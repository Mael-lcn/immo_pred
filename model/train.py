"""
train.py

Script d'entraînement principal pour le modèle d'estimation immobilière SOTA (State-of-the-Art).

Fonctionnalités :
- Gestion automatique du matériel : Nvidia (CUDA), Apple Silicon (MPS), ou CPU.
- Entraînement en Précision Mixte (AMP) pour accélérer et réduire la mémoire.
- Monitoring complet : Sauvegarde des courbes de perte (PNG) et des logs (JSON) en temps réel.
- Compatibilité avec l'architecture FT-Transformer (séparation données continues/catégorielles).

Usage :
    python train.py --epochs 20 --batch_size 16 --lr 5e-5
"""

import os
import json
import multiprocessing
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Importation de l'architecture SOTA définie dans model.py
from model import SOTARealEstateModel

# Importation des utilitaires de données
from load_data import (
    RealEstateDataset, 
    real_estate_collate_fn, 
    prepare_scaler_from_subset, 
    get_cols_config
)



# Configuration Matplotlib : Backend 'agg' pour générer des graphiques sans écran (serveur/headless)
plt.switch_backend('agg')


# ==============================================================================
# 1. UTILITAIRES DE MONITORING
# ==============================================================================
def save_monitoring(history, checkpoint_dir):
    """
    Génère et sauvegarde les fichiers de suivi de l'entraînement.
    
    Args:
        history (dict): Dictionnaire contenant les listes 'epoch' et 'loss'.
        checkpoint_dir (str): Chemin du dossier où sauvegarder les fichiers.
    """
    # 1. Sauvegarde des données brutes en JSON (pour analyse future ou reprise)
    json_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=4)

    # 2. Génération du Graphique de Convergence
    epochs = history['epoch']
    losses = history['loss']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='#1f77b4', label='Training Loss')
    
    plt.title(f"Convergence du modèle SOTA (Epoch {epochs[-1]})")
    plt.xlabel("Époques")
    plt.ylabel("Loss (Masked LogMSE)")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Sauvegarde de l'image
    plot_path = os.path.join(checkpoint_dir, "convergence_plot.png")
    plt.savefig(plot_path)
    plt.close() # Important : Libère la mémoire graphique


# ==============================================================================
# 2. FONCTION DE PERTE (LOSS)
# ==============================================================================
class MaskedLogMSELoss(nn.Module):
    """
    Fonction de perte robuste qui gère les données manquantes.
    Si une cible (Vente ou Location) est manquante (valeur 0 ou -1), elle est ignorée.
    """
    def __init__(self):
        super().__init__()
        # 'none' permet de garder l'erreur pour chaque élément du batch avant de masquer
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred_vente, pred_loc, targets, masks):
        """
        Calcule la MSE uniquement sur les cibles valides.
        
        Args:
            pred_vente (Tensor): Prédictions log prix vente [Batch, 1]
            pred_loc (Tensor): Prédictions log loyer [Batch, 1]
            targets (Tensor): Vraies valeurs (logs) [Batch, 2]
            masks (Tensor): Masque binaire (1 si présent, 0 si absent) [Batch, 2]
        
        Returns:
            Tensor: Scalaire représentant la perte moyenne.
        """
        # On concatène les deux prédictions pour matcher la forme des targets [Batch, 2]
        preds = torch.cat([pred_vente, pred_loc], dim=1)
        
        # Calcul de l'erreur quadratique
        raw_loss = self.mse(preds, targets)
        
        # Application du masque : on met à 0 l'erreur des données manquantes
        masked_loss = raw_loss * masks
        
        # On fait la moyenne uniquement sur les éléments présents
        # (masks.sum() compte le nombre de valeurs valides)
        return masked_loss.sum() / (masks.sum() + 1e-8)


# ==============================================================================
# 3. BOUCLE D'ENTRAÎNEMENT (UNE ÉPOQUE)
# ==============================================================================
def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch_index, total_epochs, use_amp=False):
    """
    Exécute une époque complète d'entraînement (Forward + Backward).
    
    Args:
        model (nn.Module): Le modèle SOTA.
        dataloader (DataLoader): Le chargeur de données PyTorch.
        optimizer (Optimizer): L'optimiseur (ex: AdamW).
        criterion (Loss): La fonction de coût.
        scaler (GradScaler): Scaler pour la précision mixte (Nvidia uniquement).
        device (torch.device): CPU, CUDA ou MPS.
        epoch_index (int): Numéro de l'époque actuelle.
        total_epochs (int): Nombre total d'époques.
        use_amp (bool): Activer ou non la précision mixte.
        
    Returns:
        float: La perte moyenne sur l'époque.
    """
    model.train() # Met le modèle en mode entraînement (active Dropout, BatchNorm...)
    
    # Barre de progression
    loop = tqdm(dataloader, desc=f"Ep {epoch_index}/{total_epochs}")
    total_loss = 0
    count = 0

    for batch in loop:
        # --- A. Transfert des données sur le GPU/MPS ---
        # non_blocking=True accélère le transfert RAM -> VRAM
        imgs = batch['images'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        mask = batch['attention_mask'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        masks = batch['masks'].to(device, non_blocking=True)
        
        # Gestion Tabulaire SOTA : Séparation Continus / Catégoriels
        # Note : On suppose ici que 'tab_data' contient les continus.
        # Si vous avez mis à jour load_data.py pour avoir 'cat_data', décommentez la ligne suivante.
        x_cont = batch['tab_data'].to(device, non_blocking=True)
        x_cat = batch['cat_data'].to(device, non_blocking=True) if 'cat_data' in batch else None

        # --- B. Forward & Backward ---
        optimizer.zero_grad(set_to_none=True) # Reset optimisé des gradients

        # Branche 1 : NVIDIA (Mixed Precision)
        if use_amp:
            with torch.amp.autocast('cuda'):
                # Appel du modèle avec la nouvelle signature
                p_vente, p_loc = model(imgs, input_ids, mask, x_cont=x_cont, x_cat=x_cat)
                loss = criterion(p_vente, p_loc, targets, masks)

            # Backpropagation avec scaling (évite underflow des gradients float16)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Branche 2 : MAC / CPU (Précision standard Float32)
        else:
            p_vente, p_loc = model(imgs, input_ids, mask, x_cont=x_cont, x_cat=x_cat)
            loss = criterion(p_vente, p_loc, targets, masks)
            
            loss.backward()
            optimizer.step()

        # --- C. Stats ---
        loss_val = loss.item()
        total_loss += loss_val
        count += 1

        # Mise à jour de la barre de progression
        loop.set_postfix(loss=f"{loss_val:.4f}", avg=f"{total_loss/count:.4f}")

    # Retourne la moyenne de la loss sur l'époque
    return total_loss / count if count > 0 else 0.0


# ==============================================================================
# 4. FONCTION PRINCIPALE (MAIN)
# ==============================================================================
def main():
    """
    Point d'entrée du script. 
    Parse les arguments, initialise le modèle et lance la boucle d'entraînement.
    """
    parser = argparse.ArgumentParser(description="Entraînement Modèle SOTA Immobilier")

    # Chemins des données
    parser.add_argument('--csv_folder', type=str, default='../output/csv/train', help="Dossier contenant les CSVs d'entraînement")
    parser.add_argument('--img_dir', type=str, default='../output/images', help="Dossier contenant les images")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_sota', help="Dossier de sauvegarde")

    # Hyperparamètres
    parser.add_argument('--batch_size', type=int, default=8, help="Taille du batch (réduire si OOM)")
    parser.add_argument('--epochs', type=int, default=20, help="Nombre d'époques")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning Rate (plus faible pour le Fine-Tuning)")
    parser.add_argument('--workers', type=int, default=max(1, multiprocessing.cpu_count()//2), help="Nombre de threads CPU pour charger les données")

    args = parser.parse_args()

    # Création du dossier de sauvegarde
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- 1. DÉTECTION DU MATÉRIEL ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True 
        print("[INIT] Mode: NVIDIA CUDA (Mixed Precision ON)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = False 
        print("[INIT] Mode: APPLE METAL (MPS)")
    else:
        device = torch.device("cpu")
        use_amp = False
        print("[INIT] Mode: CPU (Attention c'est lent)")

    print(f"[CONF] Batch={args.batch_size} | LR={args.lr} | Workers={args.workers}")

    # --- 2. PRÉPARATION DES DONNÉES ---
    print("[DATA] Analyse des fichiers CSV...")
    try:
        # On cherche un fichier CSV exemple pour calibrer les colonnes
        dummy_file = [f for f in os.listdir(args.csv_folder) if f.endswith('.csv')][0]
    except IndexError:
        print(f"[ERREUR] Aucun fichier .csv trouvé dans {args.csv_folder}")
        return

    # Récupération de la config des colonnes
    num_cols, text_cols = get_cols_config(os.path.join(args.csv_folder, dummy_file))
    
    # Préparation du scaler (Normalisation des données chiffrées)
    global_scaler = prepare_scaler_from_subset(args.csv_folder, num_cols)
    
    # Tokenizer pour le texte
    tokenizer = AutoTokenizer.from_pretrained('almanach/camembert-base')

    print("[DATA] Chargement du Dataset...")
    train_ds = RealEstateDataset(
        csv_folder=args.csv_folder,
        img_dir=args.img_dir,
        tokenizer=tokenizer,
        scaler=global_scaler,
        num_cols=num_cols,
        text_cols=text_cols,
        is_train=True # Active l'augmentation de données (Data Augmentation)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True, # Important : Mélanger les données à chaque époque
        num_workers=args.workers,
        collate_fn=real_estate_collate_fn,
        pin_memory=True
    )

    # --- 3. PRÉPARATION DU MODÈLE SOTA ---
    print("[MODEL] Initialisation de l'architecture SOTA...")
    
    # TODO : Mettre à jour cat_cardinalities
    # Exemple : cat_cardinalities=[7] si on a 1 colonne catégorielle avec 7 valeurs uniques
    current_cat_cardinalities = [] 

    model = SOTARealEstateModel(
        num_continuous=len(num_cols),
        cat_cardinalities=current_cat_cardinalities,
        img_model_name='convnext_large.fb_in1k',
        text_model_name='almanach/camembert-base',
        fusion_dim=512,
        depth=4,                # Profondeur du Transformer de fusion
        freeze_encoders=True    # On commence avec les experts gelés
    ).to(device)

    # Initialisation intelligente des biais de la dernière couche
    # Cela aide le modèle à converger plus vite en partant de la moyenne des prix
    # log(300 000) ~ 12.7  |  log(1000) ~ 7.0
    model.head_price[-1].bias.data.fill_(12.7)
    model.head_rent[-1].bias.data.fill_(7.0)

    # Optimiseur
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Fonction de coût
    criterion = MaskedLogMSELoss()

    # Scaler (uniquement utile pour Nvidia/CUDA)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Historique pour le monitoring
    history = {'epoch': [], 'loss': []}

    # --- 4. BOUCLE D'ENTRAÎNEMENT ---
    print(f"\n[TRAIN] Démarrage de l'entraînement pour {args.epochs} époques...")

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
            use_amp=use_amp
        )

        print(f"Epoch {epoch} terminée. Loss Moyenne: {avg_loss:.4f}")

        # --- Monitoring ---
        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        save_monitoring(history, args.checkpoint_dir)
        print(f"-> Monitoring mis à jour : {args.checkpoint_dir}/convergence_plot.png")

        # --- Sauvegarde ---
        # On garde tout mais seul le dernier est utile
        save_path = os.path.join(args.checkpoint_dir, f"model_ep{epoch}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"-> Checkpoint sauvegardé : {save_path}\n")

    print("[FIN] Entraînement terminé avec succès.")


if __name__ == "__main__":
    main()
