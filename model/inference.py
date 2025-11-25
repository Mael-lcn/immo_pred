"""
Prédit les prix en EUROS directement.
"""

import torch

from model import AdvancedRealEstateModel
from dataset_functional import load_images_for_batch



def predict_single_row(row, img_dir, model, tokenizer, device, num_cols, scaler):
    # Prépa manuelle d'une ligne
    id_str = str(row['id'])
    imgs = load_images_for_batch(img_dir, [id_str]).to(device) # Batch de 1

    txt = " ".join([str(row[c]) for c in row.index if c not in num_cols and c not in ['id','price']]) # Simplifié
    txt_in = tokenizer([txt], return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Note: En prod, il faut charger le scaler sauvegardé !
    nums = row[num_cols].fillna(0).values.reshape(1, -1)
    if scaler: nums = scaler.transform(nums)
    tab = torch.tensor(nums, dtype=torch.float32).to(device)

    with torch.no_grad():
        # Le modèle est en EVAL -> Il renvoie des EUROS
        p_vente, p_loc = model(imgs, txt_in['input_ids'], txt_in['attention_mask'], tab)

    return p_vente.item(), p_loc.item()
