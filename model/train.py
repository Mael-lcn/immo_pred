"""
train.py - Script d'entraînement SOTA Corrigé.

"""

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

from model import SOTARealEstateModel

from data_loader import (
    RealEstateDataset, 
    real_estate_collate_fn, 
    prepare_preprocessors, 
    get_cols_config
)



# Backend pour graphiques sans écran
plt.switch_backend('agg')


# ==============================================================================
# 1. MONITORING & SAUVEGARDE
# ==============================================================================
def save_monitoring(history, checkpoint_dir):
    json_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=4)

    epochs = history['epoch']
    losses = history['loss']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='#1f77b4', label='Training Loss')
    plt.title(f"Convergence SOTA (Epoch {epochs[-1]})")
    plt.xlabel("Époques")
    plt.ylabel("Loss (Masked LogMSE)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "convergence_plot.png"))
    plt.close()


# ==============================================================================
# 2. LOSS FUNCTION (Log-Space)
# ==============================================================================
class MaskedLogMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred_vente, pred_loc, targets, masks):
        # pred_vente/loc sont des LOGS de prix
        # targets sont aussi des LOGS de prix (fait dans le DataLoader)
        
        preds = torch.cat([pred_vente, pred_loc], dim=1) # [B, 2]
        raw_loss = self.mse(preds, targets)

        # On applique le masque (0 si donnée manquante)
        masked_loss = raw_loss * masks

        # Moyenne uniquement sur les valeurs présentes
        return masked_loss.sum() / (masks.sum() + 1e-8)


# ==============================================================================
# 3. BOUCLE D'ENTRAÎNEMENT
# ==============================================================================
def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch_index, total_epochs, use_amp=False):
    model.train()
    
    loop = tqdm(dataloader, desc=f"Ep {epoch_index}/{total_epochs}")
    total_loss = 0
    count = 0

    for batch in loop:
        # --- A. Transfert GPU ---
        # Vision
        imgs = batch['images'].to(device, non_blocking=True)
        img_masks = batch['image_masks'].to(device, non_blocking=True)
        
        # Texte
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        text_mask = batch['attention_mask'].to(device, non_blocking=True)
        
        # Tabulaire
        x_cont = batch['tab_cont'].to(device, non_blocking=True)
        x_cat = batch['tab_cat'].to(device, non_blocking=True)

        # Targets
        targets = batch['targets'].to(device, non_blocking=True)
        masks = batch['masks'].to(device, non_blocking=True)

        # --- B. Forward / Backward ---
        optimizer.zero_grad(set_to_none=True)

        # Contexte AMP (Mixed Precision)
        with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp):
            
            # Note: Le modèle doit retourner les LOGS (pas torch.exp)
            p_vente, p_loc, _ = model(
                imgs, 
                img_masks,
                input_ids, 
                text_mask, 
                x_cont, 
                x_cat
            )

            loss = criterion(p_vente, p_loc, targets, masks)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # --- C. Stats ---
        loss_val = loss.item()
        total_loss += loss_val
        count += 1
        loop.set_postfix(loss=f"{loss_val:.4f}", avg=f"{total_loss/count:.4f}")

    return total_loss / count if count > 0 else 0.0


# ==============================================================================
# 4. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_folder', type=str, default='../output/train')
    parser.add_argument('--img_dir', type=str, default='../output/filtered_images')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Hardware
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True
        print("[INIT] Mode: NVIDIA CUDA (AMP On)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = False # MPS gère mal l'AMP
        print("[INIT] Mode: APPLE METAL (MPS)")
    else:
        device = torch.device("cpu")
        use_amp = False
        print("[INIT] Mode: CPU")

    # 2. Data & Preprocessors
    print("[DATA] Calibration...")
    cont_cols, cat_cols, text_cols = get_cols_config()

    # Récupération des dims, y compris pour les binaires qui sont dans cat_dims
    scaler_obj, medians, cat_mappings, cat_dims = prepare_preprocessors(args.csv_folder, cont_cols, cat_cols)

    tokenizer = AutoTokenizer.from_pretrained('almanach/camembert-base')

    train_ds = RealEstateDataset(
        csv_folder=args.csv_folder,
        img_dir=args.img_dir,
        tokenizer=tokenizer,
        scaler=scaler_obj,
        medians=medians,
        cat_mappings=cat_mappings,
        cont_cols=cont_cols,
        cat_cols=cat_cols
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=real_estate_collate_fn,
        pin_memory=True
    )

    # 3. Model
    print(f"[MODEL] Init SOTA... (Tab Continuous: {len(cont_cols)}, Tab Categorical: {len(cat_dims)})")
    
    model = SOTARealEstateModel(
        num_continuous=len(cont_cols),
        cat_cardinalities=cat_dims, # Contient les dims de Orientation, Type ET des Binaires
        img_model_name='convnext_large.fb_in1k',
        text_model_name='almanach/camembert-base',
        fusion_dim=512,
        depth=4,
        freeze_encoders=True
    ).to(device)

    # On compile uniquement si on est sur GPU NVIDIA (Linux/Windows)
    if torch.cuda.is_available():
        print("[OPTIM] Compilation du modèle avec torch.compile (Mode: default)...")
        try:
            # 'default' est le meilleur équilibre compilation/vitesse
            # 'reduce-overhead' est plus rapide mais consomme bcp de RAM au début
            model = torch.compile(model) 
        except Exception as e:
            print(f"[WARNING] Impossible de compiler : {e}")
            print("Passage en mode standard (Eager execution).")

    # Initialisation Bias (Log Space)
    # Log(250,000) ~= 12.4
    model.head_price[-1].bias.data.fill_(12.4)
    # Log(800) ~= 6.7
    model.head_rent[-1].bias.data.fill_(6.7)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = MaskedLogMSELoss()
    scaler_amp = torch.amp.GradScaler('cuda') if use_amp else None

    # 4. Loop
    history = {'epoch': [], 'loss': []}

    print(f"\n[TRAIN] Début entraînement sur {len(train_ds)} biens.")

    for epoch in range(1, args.epochs+1):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, 
            scaler_amp, device, epoch, args.epochs, use_amp
        )

        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        save_monitoring(history, args.checkpoint_dir)

        # Sauvegarde régulière
        if epoch % 5 == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"model_ep{epoch}.pt"))
            print(f"-> Checkpoint saved: model_ep{epoch}.pt")

    print("[FIN] Terminé.")


if __name__ == "__main__":
    main()

