import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg') # Mode sans écran
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import gc

# Imports locaux
try:
    from model import SOTARealEstateModel
    from data_loader import (
        RealEstateDataset, real_estate_collate_fn, 
        prepare_preprocessors, get_cols_config
    )
except ImportError as e:
    print(f"ERREUR : Impossible d'importer model ou data_loader. {e}")
    exit(1)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='../output/train')
    parser.add_argument('--img_dir', type=str, default='../output/filtered_images')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model_ep28.pt')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Device: {device}")
    
    # 1. DATA SETUP
    print("[DATA] Préparation du sample...")
    cont_cols, cat_cols, _ = get_cols_config()
    scaler, medians, modes, cat_mappings, cat_dims = prepare_preprocessors(args.train_csv, cont_cols, cat_cols)
    tokenizer = AutoTokenizer.from_pretrained('almanach/camembert-base')

    ds = RealEstateDataset(args.train_csv, args.img_dir, tokenizer, scaler, medians, modes, cat_mappings, cont_cols, cat_cols)
    # Shuffle=True est important pour avoir un sample représentatif du dataset global
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=real_estate_collate_fn)

    # 2. MODEL LOAD
    print(f"[MODEL] Chargement {args.model_path}...")
    model = SOTARealEstateModel(len(cont_cols), cat_dims, 'convnext_large.fb_in1k', 'almanach/camembert-base', 512, 4, True).to(device)
    
    try:
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state.items()})
    except Exception as e:
        print(f"[ERREUR] Poids introuvables : {e}")
        exit(1)

    model.eval()
    model.head_price = nn.Identity() # On veut les embeddings bruts (512 dims)

    # 3. EXTRACTION (SAMPLE)
    embeddings_list = []

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    with torch.no_grad():
        for batch in tqdm(loader):
            # Optimisation mémoire GPU
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                imgs = batch['images'].to(device, non_blocking=True)
                img_masks = batch['image_masks'].to(device, non_blocking=True)
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                text_mask = batch['text_mask'].to(device, non_blocking=True)
                x_cont = batch['x_cont'].to(device, non_blocking=True)
                x_cat = batch['x_cat'].to(device, non_blocking=True)

                # Extraction
                emb, _ = model(imgs, img_masks, input_ids, text_mask, x_cont, x_cat)
            
            # Stockage CPU
            embeddings_list.append(emb.float().cpu().numpy())
            
            # Nettoyage
            del imgs, input_ids, emb

    
    X_emb = np.vstack(embeddings_list)
    print(f"[PCA] Analyse sur la matrice : {X_emb.shape}")

    # 4. CALCUL PCA
    # On calcule toutes les composantes possibles (max 512)
    pca = PCA()
    pca.fit(X_emb)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    # 5. RÉSULTATS CLÉS
    d90 = np.argmax(cumsum >= 0.90) + 1
    d95 = np.argmax(cumsum >= 0.95) + 1
    d99 = np.argmax(cumsum >= 0.99) + 1

    print("\n" + "="*40)
    print(" RÉSULTATS D'ANALYSE DE DIMENSIONNALITÉ")
    print("="*40)
    print(f"Original  : 512 dimensions")
    print(f"--> 90% d'info conservée avec : {d90} dimensions")
    print(f"--> 95% d'info conservée avec : {d95} dimensions")
    print(f"--> 99% d'info conservée avec : {d99} dimensions")
    print("="*40)

    # 6. PLOT
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumsum) + 1), cumsum, label='Variance Cumulée')
    plt.axhline(0.95, color='r', linestyle='--', label='Seuil 95%')
    plt.axvline(d95, color='r', linestyle='--', alpha=0.3)
    
    plt.scatter([d95], [cumsum[d95-1]], color='red', zorder=5)
    plt.text(d95 + 5, 0.93, f'95% @ {d95} dim', color='red', fontweight='bold')

    plt.title(f"PCA Elbow Plot (Sample N={len(X_emb)})")
    plt.xlabel("Nombre de Dimensions")
    plt.ylabel("Variance Expliquée Cumulative")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_file = "analyse_pca_resultat.png"
    plt.savefig(out_file)
    print(f"[FIN] Graphique sauvegardé sous : {out_file}")

if __name__ == "__main__":
    main()
