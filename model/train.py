"""
Script d'entraînement. Utilise le générateur streaming et la loss masquée.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer

from src.model import AdvancedRealEstateModel
from src.dataset_functional import create_streaming_generator, prepare_scaler_from_subset, get_cols_config



# --- Config ---
CSV_FOLDER = "data/csv_folder" # Dossier contenant part1.csv, part2.csv...
IMG_DIR = "data/images"
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4


class MaskedLogMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
    def forward(self, pred_vente, pred_loc, targets, masks):
        preds = torch.cat([pred_vente, pred_loc], dim=1)
        loss = self.mse(preds, targets) * masks
        return loss.sum() / (masks.sum() + 1e-8)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 1. Prépare Scaler & Config
    print("[INFO] Preparing Scaler...")
    dummy_file = os.listdir(CSV_FOLDER)[0]
    num_cols, _ = get_cols_config(os.path.join(CSV_FOLDER, dummy_file))
    global_scaler = prepare_scaler_from_subset(CSV_FOLDER, num_cols)

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # 2. Modèle
    print("[INFO] Init Model...")
    model = AdvancedRealEstateModel(
        img_model_name='convnext_large.fb_in1k',
        text_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        tab_input_dim=len(num_cols),
        fusion_dim=512,
        freeze_encoders=True # On gèle les experts
    ).to(device)

    # --- ASTUCE BIAIS LOG ---
    # On force la sortie à démarrer autour de 12.5 (Vente) et 7.0 (Loc)
    print("[INFO] Initializing Bias for Log Space...")
    # TO DO init avec log mean des biens pour etre prrécis
    model.head_price[-1].bias.data.fill_(12.5) 
    model.head_rent[-1].bias.data.fill_(7.0)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = MaskedLogMSELoss()
    scaler = torch.amp.GradScaler('cuda')

    # 3. Boucle
    for epoch in range(EPOCHS):
        model.train()
        train_gen = create_streaming_generator(CSV_FOLDER, IMG_DIR, tokenizer, BATCH_SIZE, global_scaler, True)

        loop = tqdm(train_gen, desc=f"Ep {epoch+1}") # Pas de 'total' car streaming
        total_loss = 0
        count = 0

        for batch in loop:
            # Move to GPU
            imgs = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            tab = batch['tab_data'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                # Le modèle renvoie des LOGS ici (car training=True)
                p_vente, p_loc = model(imgs, input_ids, mask, tab)
                loss = criterion(p_vente, p_loc, targets, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            count += 1
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Avg Loss: {total_loss/count:.4f}")
        torch.save(model.state_dict(), f"checkpoints/model_ep{epoch}.pt")


if __name__ == "__main__":
    main()
