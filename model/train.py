import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# --- IMPORTS LOCAUX ---
# On charge tes classes personnalisées
from model import SOTARealEstateModel
from data_loader import (
    RealEstateDataset, 
    real_estate_collate_fn, 
    prepare_preprocessors, 
    get_cols_config
)

# Force matplotlib à ne pas chercher d'écran (utile sur serveur)
plt.switch_backend('agg')


# ==============================================================================
# 1. OUTILS DE MONITORING (Graphiques & Historique)
# ==============================================================================
def save_monitoring(history, checkpoint_dir):
    """
    Sauvegarde l'historique en JSON et trace les courbes de Loss.
    Permet de vérifier visuellement s'il y a de l'Overfitting (si Val remonte).
    """
    # 1. Sauvegarde JSON
    json_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=4)

    # 2. Tracé du Graphique
    epochs = history['epoch']
    train_loss = history['train_loss']
    val_loss = history.get('val_loss', [])

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, marker='o', label='Train Loss', color='#1f77b4') # Bleu
    if val_loss:
        plt.plot(epochs, val_loss, marker='x', linestyle='--', label='Val Loss', color='#ff7f0e') # Orange
    
    plt.title("Convergence SOTA : Train vs Validation")
    plt.xlabel("Époques")
    plt.ylabel("Loss (Masked LogMSE)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(os.path.join(checkpoint_dir, "convergence_plot.png"))
    plt.close()


# ==============================================================================
# 2. FONCTION DE PERTE (Masked Log MSE)
# ==============================================================================
class MaskedLogMSELoss(nn.Module):
    """
    Calcule la MSE sur les Logs des prix.
    Ignore intelligemment la tête inutile (ex: ignore tête Location si c'est une Vente).
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none') # On veut la perte par élément pour appliquer le masque

    def forward(self, pred_vente, pred_loc, targets, masks):
        # pred_vente : [Batch, 1] (Prédiction du modèle)
        # targets    : [Batch, 2] (Target Vente, Target Loc)
        # masks      : [Batch, 2] (1 si la donnée existe, 0 sinon)

        target_vente = targets[:, 0].unsqueeze(1)
        target_loc   = targets[:, 1].unsqueeze(1)
        mask_vente   = masks[:, 0].unsqueeze(1)
        mask_loc     = masks[:, 1].unsqueeze(1)

        # Calcul de l'erreur brute * Masque (Mise à zéro si pas concerné)
        loss_vente = self.mse(pred_vente, target_vente) * mask_vente
        loss_loc   = self.mse(pred_loc, target_loc) * mask_loc

        # Somme des erreurs / Nombre de vrais exemples (pour éviter la dilution par les zéros)
        total_loss = loss_vente.sum() + loss_loc.sum()
        denominator = mask_vente.sum() + mask_loc.sum()
        
        return total_loss / (denominator + 1e-8)


# ==============================================================================
# 3. BOUCLE D'ENTRAÎNEMENT (Le modèle APPREND)
# ==============================================================================
def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch, total_epochs, use_amp):
    model.train() # <--- IMPORTANT : Active Dropout & BatchNorm

    loop = tqdm(dataloader, desc=f"Ep {epoch}/{total_epochs} [TRAIN]")
    total_loss = 0
    count = 0

    for batch in loop:
        # --- A. Chargement GPU ---
        # Utilisation de non_blocking=True pour accélérer le transfert RAM -> VRAM
        imgs = batch['images'].to(device, non_blocking=True)
        img_masks = batch['image_masks'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        text_mask = batch['text_mask'].to(device, non_blocking=True)
        x_cont = batch['x_cont'].to(device, non_blocking=True)
        x_cat = batch['x_cat'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        masks = batch['masks'].to(device, non_blocking=True)

        # --- B. Forward & Backward ---
        optimizer.zero_grad(set_to_none=True) # Reset gradients

        # AMP (Automatic Mixed Precision) : Boost vitesse sur GPU NVIDIA
        with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp):
            p_vente, p_loc = model(
                images=imgs, 
                image_masks=img_masks, 
                input_ids=input_ids, 
                text_mask=text_mask, 
                x_cont=x_cont, 
                x_cat=x_cat
            )
            loss = criterion(p_vente, p_loc, targets, masks)

        # Gestion du gradient (avec ou sans scaler AMP)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # --- C. Stats ---
        total_loss += loss.item()
        count += 1
        loop.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss/count:.4f}")

    return total_loss / count if count > 0 else 0.0


# ==============================================================================
# 4. BOUCLE DE VALIDATION (Le modèle est TESTÉ)
# ==============================================================================
def validate(model, dataloader, criterion, device, use_amp):
    model.eval() # <--- IMPORTANT : Désactive Dropout, fige BatchNorm
    
    loop = tqdm(dataloader, desc="[VAL]")
    total_loss = 0
    count = 0

    with torch.no_grad(): # <--- IMPORTANT : Coupe l'enregistrement des gradients (économise VRAM)
        for batch in loop:
            imgs = batch['images'].to(device, non_blocking=True)
            img_masks = batch['image_masks'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            text_mask = batch['text_mask'].to(device, non_blocking=True)
            x_cont = batch['x_cont'].to(device, non_blocking=True)
            x_cat = batch['x_cat'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            masks = batch['masks'].to(device, non_blocking=True)

            with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp):
                p_vente, p_loc = model(imgs, img_masks, input_ids, text_mask, x_cont, x_cat)
                loss = criterion(p_vente, p_loc, targets, masks)

            total_loss += loss.item()
            count += 1
            loop.set_postfix(val_loss=f"{total_loss/count:.4f}")

    return total_loss / count if count > 0 else 0.0


# ==============================================================================
# 5. MAIN (Orchestration)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    # Dossiers séparés physiquement (générés par ton script de split)
    parser.add_argument('--train_csv', type=str, default='../output/train')
    parser.add_argument('--val_csv', type=str, default='../output/val')
    # Les images sont généralement dans un dossier commun
    parser.add_argument('--img_dir', type=str, default='../output/filtered_images')
    
    # Paramètres d'entraînement
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=16) # Ajuster selon VRAM (16 ou 32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4) # 0.0001 est standard pour les Transformers
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Détection Hardware
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True
        print("[INIT] Mode: NVIDIA CUDA (AMP On)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = False 
        print("[INIT] Mode: APPLE METAL (MPS)")
    else:
        device = torch.device("cpu")
        use_amp = False
        print("[INIT] Mode: CPU")

    # 2. Calibration des Données (CRITIQUE : UNIQUEMENT SUR TRAIN)
    # On apprend "le monde" à travers le Train Set uniquement pour ne pas tricher.
    print("[DATA] Calibration des scalers et vocabulaires sur le TRAIN...")
    cont_cols, cat_cols, text_cols = get_cols_config()
    
    # prepare_preprocessors renvoie les OBJETS calibrés (scaler, modes, mappings)
    scaler_obj, medians, modes, cat_mappings, cat_dims = prepare_preprocessors(args.train_csv, cont_cols, cat_cols)
    
    tokenizer = AutoTokenizer.from_pretrained('almanach/camembert-base')

    # 3. Création des Datasets & Loaders
    print(f"[DATA] Chargement Train : {args.train_csv}")
    train_ds = RealEstateDataset(
        df_or_folder=args.train_csv, img_dir=args.img_dir, tokenizer=tokenizer,
        scaler=scaler_obj, medians=medians, modes=modes, cat_mappings=cat_mappings,
        cont_cols=cont_cols, cat_cols=cat_cols
    )
    
    print(f"[DATA] Chargement Validation : {args.val_csv}")
    # Pour la Validation, on réutilise les objets calibrés ci-dessus (scaler_obj, etc.)
    val_ds = RealEstateDataset(
        df_or_folder=args.val_csv, img_dir=args.img_dir, tokenizer=tokenizer,
        scaler=scaler_obj, medians=medians, modes=modes, cat_mappings=cat_mappings,
        cont_cols=cont_cols, cat_cols=cat_cols
    )

    # Note : shuffle=True pour Train, shuffle=False pour Val
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.workers, collate_fn=real_estate_collate_fn, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.workers, collate_fn=real_estate_collate_fn, pin_memory=True
    )

    # 4. Initialisation du Modèle
    print(f"[MODEL] Init SOTA... (Tab Continuous: {len(cont_cols)}, Tab Categorical: {len(cat_dims)})")
    model = SOTARealEstateModel(
        num_continuous=len(cont_cols),
        cat_cardinalities=cat_dims,
        img_model_name='convnext_large.fb_in1k', # Très bon modèle vision
        text_model_name='almanach/camembert-base',
        fusion_dim=512,
        depth=4,
        freeze_encoders=True # On gèle CamemBERT et ConvNext au début pour stabiliser
    ).to(device)

    # Astuce SOTA : Initialisation intelligente des biais de sortie
    # On initialise le bias pour qu'il prédise le prix moyen dès le début.
    # Log(250,000) ~= 12.4
    model.head_price[-1].bias.data.fill_(12.4)
    # Log(800) ~= 6.7
    model.head_rent[-1].bias.data.fill_(6.7)

    # Compilation PyTorch 2.0 (Accélération gratuite si dispo)
    if torch.cuda.is_available():
        try: 
            model = torch.compile(model)
            print("[OPTIM] Torch.compile activé.")
        except: 
            pass

    # Optimiseur & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = MaskedLogMSELoss()
    scaler_amp = torch.amp.GradScaler('cuda') if use_amp else None

    # 5. Boucle Principale
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    print(f"\n[TRAIN] Démarrage : {len(train_ds)} train samples | {len(val_ds)} val samples.")

    for epoch in range(1, args.epochs+1):
        # A. Phase d'Apprentissage
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, 
            scaler_amp, device, epoch, args.epochs, use_amp
        )
        
        # B. Phase de Validation (Conditionnelle : Seulement >= 20)
        val_loss = None # On initialise à None

        if epoch > 20:
            val_loss = validate(model, val_loader, criterion, device, use_amp)
            print(f" -> Ep {epoch}: Train={train_loss:.4f} | Val={val_loss:.4f}")
        else:
            print(f" -> Ep {epoch}: Train={train_loss:.4f} | Val=(Skipped)")

        # C. Logs (On met une valeur placeholder si pas de validation pour ne pas casser les graphes)
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        # Si pas de validation, on remet la dernière connue ou None, ici on met None pour l'ignorer
        history['val_loss'].append(val_loss if val_loss is not None else 0.0)
        save_monitoring(history, args.checkpoint_dir)

        # D. Checkpointing Intelligent
        # On ne sauvegarde 'best_model.pt' QUE si on bat le record de validation
        if (epoch > 20) and (val_loss < best_val_loss):
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pt"))
            print(f"    *** NEW RECORD! best_model.pt sauvegardé (Val: {val_loss:.4f}) ***")

        # Sauvegarde périodique (Backups de sécurité tous les 5 epochs)
        elif epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"model_ep{epoch}.pt"))

    print("[FIN] Entraînement terminé.")


if __name__ == "__main__":
    main()
