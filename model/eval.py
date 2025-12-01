"""
Script complet d'évaluation (Inférence) - Version SOTA
Compatible : Mac (MPS), Nvidia (CUDA), CPU.
Fonctionnalités :
- Charge un checkpoint (.pth) de l'architecture SOTA
- Gère la séparation Continus/Catégoriels
- Calcule MAE, RMSE, R^2 (sur les prix réels en €)
- Génère des graphiques "Prédiction vs Réalité"
"""

import os
import argparse, multiprocessing
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- IMPORT DU NOUVEAU MODÈLE SOTA ---
from model import SOTARealEstateModel
from load_data import (
    RealEstateDataset, 
    real_estate_collate_fn, 
    prepare_scaler_from_subset, 
    get_cols_config
)



# --- 1. FONCTION D'INFÉRENCE ---
def run_inference(model, dataloader, device):
    """
    Parcourt le dataset de test et récupère les prédictions et les cibles.
    Le modèle SOTA gère lui-même la conversion Log -> Euros en mode eval().
    """
    model.eval()

    # Listes pour stocker les résultats
    preds_vente, targets_vente = [], []
    preds_loc, targets_loc = [], []

    print(f"[INFO] Démarrage de l'inférence sur {len(dataloader)} batchs...")

    with torch.no_grad(): # Désactive le calcul de gradient (économise mémoire/temps)
        for batch in tqdm(dataloader, desc="Évaluation"):
            # 1. Transfert sur GPU/MPS
            imgs = batch['images'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            mask = batch['attention_mask'].to(device, non_blocking=True)
            
            # --- MODIFICATION SOTA : Séparation des entrées ---
            # tab_data contient les données continues
            x_cont = batch['tab_data'].to(device, non_blocking=True)
            # cat_data contient les données catégorielles (si elles existent)
            x_cat = batch['cat_data'].to(device, non_blocking=True) if 'cat_data' in batch else None

            targets = batch['targets'].to(device, non_blocking=True) # [batch, 2]
            masks = batch['masks'].to(device, non_blocking=True)     # [batch, 2]

            # 2. Prédiction du modèle
            # Le modèle SOTA en mode eval() renvoie déjà des Euros (pas des logs)
            p_vente_euro, p_loc_euro = model(imgs, input_ids, mask, x_cont=x_cont, x_cat=x_cat)

            # 3. Récupération des valeurs (déjà en Euros côté modèle)
            curr_p_vente = p_vente_euro.cpu().numpy().flatten()
            curr_p_loc = p_loc_euro.cpu().numpy().flatten()

            # 4. Conversion des Targets (qui elles sont toujours en log dans le dataset)
            curr_t_vente = torch.exp(targets[:, 0]).cpu().numpy().flatten()
            curr_t_loc = torch.exp(targets[:, 1]).cpu().numpy().flatten()
            
            mask_vente = masks[:, 0].cpu().numpy().astype(bool)
            mask_loc = masks[:, 1].cpu().numpy().astype(bool)

            # 5. Stockage (Uniquement si la donnée existe)
            if mask_vente.any():
                preds_vente.extend(curr_p_vente[mask_vente])
                targets_vente.extend(curr_t_vente[mask_vente])
            
            if mask_loc.any():
                preds_loc.extend(curr_p_loc[mask_loc])
                targets_loc.extend(curr_t_loc[mask_loc])

    return (np.array(preds_vente), np.array(targets_vente)), \
           (np.array(preds_loc), np.array(targets_loc))


# --- 2. CALCUL MÉTRIQUES ET AFFICHAGE ---
def print_metrics_and_plot(preds, targets, category_name, save_dir="evaluation_results"):
    if len(preds) == 0:
        print(f"\n--- {category_name} : Aucune donnée trouvée ---")
        return

    # On crée un masque pour garder seulement les valeurs finies
    mask_clean = np.isfinite(preds) & np.isfinite(targets)
    
    # On filtre
    preds_clean = preds[mask_clean]
    targets_clean = targets[mask_clean]

    # On vérifie combien on a rejeté
    rejected = len(preds) - len(preds_clean)
    if rejected > 0:
        print(f"[ATTENTION] {rejected} valeurs aberrantes (Inf/NaN) ignorées pour {category_name}.")

    if len(preds_clean) == 0:
        print(f"[ERREUR] Toutes les prédictions sont invalides pour {category_name}.")
        return

    # Création du dossier de résultats
    os.makedirs(save_dir, exist_ok=True)

    # Calcul des métriques sur les données nettoyées
    mae = mean_absolute_error(targets_clean, preds_clean)
    mse = mean_squared_error(targets_clean, preds_clean)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_clean, preds_clean)

    # Affichage Console
    print(f"\n" + "="*40)
    print(f" RÉSULTATS : {category_name}")
    print(f"="*40)
    print(f"Nombre d'échantillons valides : {len(preds_clean)} (Total: {len(preds)})")
    print(f"MAE (Erreur Moyenne)  : {mae:,.2f} €")
    print(f"RMSE (Erreur Quadr.)  : {rmse:,.2f} €")
    print(f"R² Score              : {r2:.4f}")
    print(f"="*40)

    # Création du Graphique (Scatter Plot)
    plt.figure(figsize=(8, 8))

    # Nuage de points
    plt.scatter(targets_clean, preds_clean, alpha=0.5, s=15, c='blue', edgecolors='none', label='Prédictions')

    # Ligne idéale (y = x)
    min_val = min(targets_clean.min(), preds_clean.min())
    max_val = max(targets_clean.max(), preds_clean.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Parfait (y=x)')

    plt.title(f"Prédictions vs Réalité - {category_name}\nR² = {r2:.3f} | MAE = {mae:.0f}€")
    plt.xlabel("Prix Réel (€)")
    plt.ylabel("Prix Prédit (€)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Sauvegarde
    filename = os.path.join(save_dir, f"plot_{category_name.lower()}.png")
    plt.savefig(filename)
    print(f"[INFO] Graphique sauvegardé sous : {filename}")
    plt.close()


# --- 3. MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Évaluation Modèle SOTA Immobilier")

    # Arguments
    parser.add_argument('--train_csv_folder', type=str, default='../output/csv/train', help="Dossier Train (pour calibrer le Scaler)")
    parser.add_argument('--test_csv_folder', type=str, default='../output/csv/test', help="Dossier Test (données à évaluer)")
    parser.add_argument('--img_dir', type=str, default='../output/eval', help="Dossier images")
    parser.add_argument('--model_path', type=str, default='checkpoints_sota/model_ep20.pt', help="Chemin vers le fichier .pth")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=max(1, multiprocessing.cpu_count()//2))

    args = parser.parse_args()

    # 1. Détection du Matériel
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("[INIT] Mode: NVIDIA CUDA Detected")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INIT] Mode: APPLE METAL (MPS) Detected")
    else:
        device = torch.device("cpu")
        print("[INIT] Mode: CPU (Attention à la lenteur)")

    # 2. Préparation du Scaler (Data Leakage Prevention)
    print("[INIT] Calibration du Scaler sur les données Train...")
    try:
        dummy_file = [f for f in os.listdir(args.train_csv_folder) if f.endswith('.csv')][0]
    except IndexError:
        print("[ERREUR] Dossier Train vide ou introuvable.")
        return

    num_cols, text_cols = get_cols_config(os.path.join(args.train_csv_folder, dummy_file))
    global_scaler = prepare_scaler_from_subset(args.train_csv_folder, num_cols)
    
    tokenizer = AutoTokenizer.from_pretrained('almanach/camembert-base')

    # 3. Chargement du Dataset de TEST
    print("[INIT] Chargement du Dataset de TEST...")
    test_ds = RealEstateDataset(
        csv_folder=args.test_csv_folder,
        img_dir=args.img_dir,
        tokenizer=tokenizer,
        scaler=global_scaler, # Utilise le scaler calibré sur Train
        num_cols=num_cols,
        text_cols=text_cols,
        is_train=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=real_estate_collate_fn,
        pin_memory=True
    )

    # 4. Chargement du Modèle SOTA
    print(f"[INIT] Chargement du modèle depuis : {args.model_path}")
    
    # --- MODIFICATION SOTA : Config Catégorielle ---
    # TODO : Mettre à jour cat_cardinalities comme dans train.py
    current_cat_cardinalities = [] # Ex: [7] si DPE

    model = SOTARealEstateModel(
        num_continuous=len(num_cols),
        cat_cardinalities=current_cat_cardinalities,
        img_model_name='convnext_large.fb_in1k',
        text_model_name='almanach/camembert-base',
        fusion_dim=512,
        depth=4,
        freeze_encoders=True # Pas d'impact en eval
    ).to(device)

    # Chargement sécurisé des poids
    checkpoint = torch.load(args.model_path, map_location=device)
    # Gestion de la structure du checkpoint
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    # 5. Exécution
    (preds_vente, targets_vente), (preds_loc, targets_loc) = run_inference(model, test_loader, device)

    # 6. Résultats
    print_metrics_and_plot(preds_vente, targets_vente, "VENTE")
    print_metrics_and_plot(preds_loc, targets_loc, "LOCATION")
    
    print("\n[FIN] Évaluation terminée.")

if __name__ == "__main__":
    main()
