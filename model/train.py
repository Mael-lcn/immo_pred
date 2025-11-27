"""
Script d'entraînement compatible Mac (MPS) et Nvidia (CUDA).
Gère automatiquement le device et la précision mixte.
"""

import os
import multiprocessing
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from model import AdvancedRealEstateModel
from load_data import (
    RealEstateDataset, 
    real_estate_collate_fn, 
    prepare_scaler_from_subset, 
    get_cols_config
)



# --- 1. LOSS FUNCTION ---
class MaskedLogMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred_vente, pred_loc, targets, masks):
        # Concaténation des prédictions (Batch, 2)
        preds = torch.cat([pred_vente, pred_loc], dim=1)
        # Calcul de l'erreur seulement là où le masque vaut 1
        loss = self.mse(preds, targets) * masks
        # Somme divisée par le nombre de vraies valeurs (évite division par 0)
        return loss.sum() / (masks.sum() + 1e-8)


# --- 2. FONCTION D'ENTRAÎNEMENT (UNE ÉPOQUE) ---
def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch_index, total_epochs, use_amp=False):
    model.train()
    
    loop = tqdm(dataloader, desc=f"Ep {epoch_index}/{total_epochs}")
    total_loss = 0
    count = 0

    for batch in loop:
        # Transfert vers le Device (GPU Mac ou Nvidia)
        # non_blocking=True accélère le transfert si pin_memory=True dans le loader
        imgs = batch['images'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        mask = batch['attention_mask'].to(device, non_blocking=True)
        tab = batch['tab_data'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        masks = batch['masks'].to(device, non_blocking=True)

        # Reset des gradients (set_to_none est plus rapide)
        optimizer.zero_grad(set_to_none=True)

        # --- CAS 1 : NVIDIA (Mixed Precision activé) ---
        if use_amp:
            with torch.amp.autocast('cuda'):
                p_vente, p_loc = model(imgs, input_ids, mask, tab)
                loss = criterion(p_vente, p_loc, targets, masks)
            
            # Backprop avec scaling pour éviter underflow
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # --- CAS 2 : MAC / CPU (Float32 standard) ---
        else:
            # MPS (Mac) n'aime pas toujours l'autocast, on reste en float32 (très rapide sur M1/M2)
            p_vente, p_loc = model(imgs, input_ids, mask, tab)
            loss = criterion(p_vente, p_loc, targets, masks)
            
            loss.backward()
            optimizer.step()

        # Stats
        loss_val = loss.item()
        total_loss += loss_val
        count += 1
        
        loop.set_postfix(loss=loss_val, avg=total_loss/count)

    return total_loss / count if count > 0 else 0.0


# --- 3. MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Entraînement Multimodal Immobilier")
    
    # Chemins par défaut (Ajuste selon ton arborescence)
    parser.add_argument('--csv_folder', type=str, default='../output/csv/train', help="Dossier contenant les CSVs d'entraînement")
    parser.add_argument('--img_dir', type=str, default='../../data/images', help="Dossier racine des images")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Sauvegarde des modèles")

    # Hyperparamètres
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=max(1, multiprocessing.cpu_count()//2), help="Nb de process CPU pour charger les données (0 sur Windows, 2-4 sur Mac/Linux)")

    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- DÉTECTION DU MATÉRIEL (Mac vs Nvidia vs CPU) ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True # On active l'accélération Nvidia
        print("[INFO] Mode: NVIDIA CUDA (Mixed Precision ON)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = False # On désactive AMP sur Mac pour la stabilité
        print("[INFO] Mode: APPLE METAL (MPS) (Mixed Precision OFF)")
    else:
        device = torch.device("cpu")
        use_amp = False
        print("[INFO] Mode: CPU (Attention c'est lent)")

    print(f"[INFO] Config: Batch={args.batch_size} | LR={args.lr} | Workers={args.workers}")

    # --- PRÉPARATION DONNÉES ---
    print("[INFO] Analyse des données...")
    # On trouve un fichier CSV exemple pour lire les colonnes
    try:
        dummy_file = [f for f in os.listdir(args.csv_folder) if f.endswith('.csv')][0]
    except IndexError:
        print(f"[ERREUR] Aucun fichier .csv trouvé dans {args.csv_folder}")
        return

    num_cols, text_cols = get_cols_config(os.path.join(args.csv_folder, dummy_file))
    
    # Entraînement du Scaler
    global_scaler = prepare_scaler_from_subset(args.csv_folder, num_cols)
    
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Création du Dataset
    print("[INFO] Chargement du Dataset...")
    train_ds = RealEstateDataset(
        csv_folder=args.csv_folder,
        img_dir=args.img_dir,
        tokenizer=tokenizer,
        scaler=global_scaler,
        num_cols=num_cols,
        text_cols=text_cols,
        is_train=True
    )

    # Création du DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,                # Mélange important pour l'apprentissage
        num_workers=args.workers,    # Charge les images en parallèle
        collate_fn=real_estate_collate_fn, # Gère le padding des images
        pin_memory=True              # Accélère le transfert RAM -> VRAM
    )

    # --- PRÉPARATION MODÈLE ---
    print("[INFO] Initialisation du modèle...")
    model = AdvancedRealEstateModel(
        img_model_name='convnext_large.fb_in1k',
        text_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        tab_input_dim=len(num_cols),
        fusion_dim=512,
        freeze_encoders=True 
    ).to(device)

    # Initialisation des biais pour aider la convergence (Log prices moyens)
    # Vente ~ 270k (ln ~12.5), Loc ~ 1100 (ln ~7.0)
    model.head_price[-1].bias.data.fill_(12.5)
    model.head_rent[-1].bias.data.fill_(7.0)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = MaskedLogMSELoss()
    
    # Le Scaler n'est instancié que si on utilise AMP (Nvidia)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # --- BOUCLE D'ENTRAÎNEMENT ---
    print(f"[INFO] Démarrage de l'entraînement pour {args.epochs} époques...")

    for epoch in range(1, args.epochs+1):
        avg_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epoch_index=epoch,
            total_epochs=args.epochs,
            use_amp=use_amp # Passe l'info sur le mode (Mac/PC)
        )

        print(f"Epoch {epoch} terminée. Loss Moyenne: {avg_loss:.4f}")
        
        # Sauvegarde
        save_path = os.path.join(args.checkpoint_dir, f"model_ep{epoch}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"-> Checkpoint sauvegardé : {save_path}\n")

if __name__ == "__main__":
    main()
