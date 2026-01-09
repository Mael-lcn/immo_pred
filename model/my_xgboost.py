import os
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import torch.multiprocessing
import multiprocessing
import types
import matplotlib
# Force l'utilisation d'un backend sans interface graphique (idéal pour CLI/Serveur)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import gc # Garbage Collector pour le nettoyage mémoire
from sklearn.decomposition import PCA 

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- CONFIGURATION ---
sns.set(style="whitegrid")
warnings.filterwarnings('ignore')

# --- CORRECTIFS SYSTÈME ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.multiprocessing.set_sharing_strategy('file_system')

# --- IMPORTS LOCAUX ---
try:
    from model import SOTARealEstateModel
    from data_loader import (
        RealEstateDataset, 
        real_estate_collate_fn, 
        prepare_preprocessors, 
        get_cols_config
    )
except ImportError as e:
    print(f"ERREUR D'IMPORT : {e}")
    exit(1)


# ==============================================================================
# 1. OUTIL D'ANALYSE DATA SCIENCE & SAUVEGARDE
# ==============================================================================
def evaluate_and_save(model, X_test, y_test_log, feature_names, label, output_dir):
    """
    Génère les plots, les métriques et les sauvegarde proprement dans output_dir.
    """
    print(f"\n{'='*20} GÉNÉRATION RAPPORT : {label} {'='*20}")
    
    # Création du dossier spécifique (ex: results/vente)
    save_path = os.path.join(output_dir, label.lower())
    os.makedirs(save_path, exist_ok=True)
    print(f"[IO] Sauvegarde des résultats dans : {save_path}")

    # --- 1. Prédictions & Inverse Log ---
    preds_log = model.predict(X_test)
    y_true = np.expm1(y_test_log)
    y_pred = np.expm1(preds_log)
    
    # --- 2. Calcul des Métriques ---
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Sauvegarde Metrics TXT
    with open(os.path.join(save_path, "metrics.txt"), "w") as f:
        f.write(f"=== RAPPORT {label} ===\n")
        f.write(f"R2 Score : {r2:.5f}\n")
        f.write(f"MAE      : {mae:,.2f} eur\n")
        f.write(f"RMSE     : {rmse:,.2f} eur\n")
        f.write(f"Nb Test  : {len(y_true)}\n")
    
    print(f"   > Metrics sauvegardées.")

    # --- 3. Export CSV (Crucial pour analyse manuelle) ---
    df_res = pd.DataFrame({
        'Reel': y_true,
        'Predit': y_pred,
        'Erreur': y_pred - y_true,
        'Erreur_Abs': np.abs(y_pred - y_true),
        'Erreur_Rel_%': (np.abs(y_pred - y_true) / y_true) * 100
    })
    
    csv_path = os.path.join(save_path, "predictions_full.csv")
    df_res.sort_values("Erreur_Abs", ascending=False).to_csv(csv_path, index=False)
    print(f"   > CSV détaillé sauvegardé (trié par erreur).")

    # --- 4. Dashboard Graphique ---
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(f"Analyse Performance - {label} (R²={r2:.3f})", fontsize=18)

    # A. Calibration
    sns.scatterplot(x='Reel', y='Predit', data=df_res, alpha=0.4, edgecolor=None, ax=axes[0,0])
    min_val, max_val = y_true.min(), y_true.max()
    axes[0,0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0,0].set_title("Réel vs Prédit")
    axes[0,0].set_xlabel("Prix Réel")
    axes[0,0].set_ylabel("Prix Prédit")
    
    # B. Distribution Erreurs
    sns.histplot(df_res['Erreur'], kde=True, ax=axes[0,1], color='orange', bins=50)
    axes[0,1].set_title("Distribution des Erreurs (€)")
    axes[0,1].axvline(0, color='r', linestyle='--')

    # C. Hétéroscédasticité
    sns.scatterplot(x='Predit', y='Erreur', data=df_res, alpha=0.4, ax=axes[1,0])
    axes[1,0].axhline(0, color='r', linestyle='--')
    axes[1,0].set_title("Résidus vs Valeur Prédite (Biais sur biens chers ?)")

    # D. Feature Importance (Gain XGBoost)
    if feature_names is None:
        feature_names = [f"F_{i}" for i in range(X_test.shape[1])]
    
    # Sécurité taille feature names
    if len(feature_names) != X_test.shape[1]:
        feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:20] # Top 20
    
    sns.barplot(
        x=importance[indices], 
        y=[feature_names[i] for i in indices],
        ax=axes[1,1], palette="viridis"
    )
    axes[1,1].set_title("Top 20 Feature Importance (Gain)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_path, "dashboard_performance.png"), dpi=150)
    plt.close() # Libère la mémoire
    print(f"   > Dashboard PNG sauvegardé.")

    # --- 5. SHAP ---
    print(f"   > Calcul SHAP en cours...")
    try:
        # Sampling pour rapidité
        sample_size = min(1000, X_test.shape[0])
        idx = np.random.choice(X_test.shape[0], sample_size, replace=False)
        X_sample = X_test[idx]
        X_sample_df = pd.DataFrame(X_sample, columns=feature_names)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample_df)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample_df, show=False)
        plt.title(f"SHAP Summary - {label}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "shap_summary.png"), dpi=150)
        plt.close()
        print(f"   > SHAP PNG sauvegardé.")
    except Exception as e:
        print(f"[WARN] SHAP échoué : {e}")

    return df_res

# ==============================================================================
# 2. EXTRACTION FEATURES (MODIFIÉ POUR MEMOIRE & BATCHING)
# ==============================================================================
def extract_dataset_features(loader, model, device, use_nn):
    """
    Cette fonction est le cœur de l'approche hybride.
    Elle parcourt le dataset par petits paquets (batchs).
    Si NN est activé : elle envoie le batch au GPU, récupère les embeddings (512 colonnes),
    les rapatrie sur CPU, et vide le GPU immédiatement.
    """
    features_list = []
    targets_list = []
    masks_list = []

    if use_nn and model is not None:
        model.eval()

    desc = "Extraction (HYBRIDE & IMAGES)" if use_nn else "Extraction (TABULAIRE)"
    
    # [IMPORTANT] Nettoyage préventif
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # On désactive le calcul des gradients pour économiser massivement la mémoire
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            
            # --- A. Partie Tabulaire (Légère, sur CPU) ---
            x_cont = batch['x_cont'] 
            x_cat = batch['x_cat']
            
            # On concatène les données chiffrées et catégorielles
            batch_features = torch.cat([x_cont, x_cat.float()], dim=1).cpu().numpy()

            # --- B. Partie NN (Lourde, sur GPU) ---
            if use_nn and model is not None:
                # [OPTIMISATION] Utilisation de l'AMP (Automatic Mixed Precision)
                # Cela réduit la consommation VRAM par 2 en utilisant du float16 là où c'est possible
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # 1. Chargement GPU du Batch uniquement
                    imgs = batch['images'].to(device, non_blocking=True)
                    img_masks = batch['image_masks'].to(device, non_blocking=True)
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    text_mask = batch['text_mask'].to(device, non_blocking=True)
                    x_cont_gpu = batch['x_cont'].to(device, non_blocking=True)
                    x_cat_gpu = batch['x_cat'].to(device, non_blocking=True)

                    # 2. Passage dans le modèle pour obtenir le vecteur fusionné (Embedding)
                    # Le modèle ne prédit pas un prix ici, il sort juste les "caractéristiques" (ex: 512 dims)
                    embeddings, _ = model(
                        images=imgs, image_masks=img_masks, 
                        input_ids=input_ids, text_mask=text_mask, 
                        x_cont=x_cont_gpu, x_cat=x_cat_gpu
                    )
                
                # 3. Rapatriement immédiat sur CPU et conversion en numpy
                embeddings_cpu = embeddings.float().cpu().numpy()
                
                # 4. Ajout des colonnes d'embeddings aux features tabulaires
                # C'est ici qu'on "crée les colonnes" supplémentaires pour XGBoost
                batch_features = np.hstack([batch_features, embeddings_cpu])

                # [CRITIQUE] Nettoyage immédiat de la VRAM pour le prochain batch
                del imgs, img_masks, input_ids, text_mask, embeddings, x_cont_gpu, x_cat_gpu
                # Optionnel : force le vidage cache si vraiment limite en RAM
                # torch.cuda.empty_cache() 

            # --- C. Stockage dans la liste (RAM Système) ---
            features_list.append(batch_features)
            targets_list.append(batch['targets'].numpy())
            masks_list.append(batch['masks'].numpy())

    # Une fois la boucle finie, on empile tout pour faire une grosse matrice
    return np.vstack(features_list), np.vstack(targets_list), np.vstack(masks_list)

# ==============================================================================
# 3. ENTRAÎNEMENT CORE
# ==============================================================================
def train_xgb_core(X_train, y_train, X_test, y_test, label):
    print(f"[{label}] Training sur {len(X_train)} samples...")

    # 
    reg = xgb.XGBRegressor(
        n_estimators=8000,
        learning_rate=0.01,
        max_depth=9,
        subsample=0.8,
        colsample_bytree=0.4,
        min_child_weight=3,
        objective='reg:absoluteerror',
        n_jobs=-1,
        early_stopping_rounds=150,
        tree_method='hist'
    )

    reg.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=1000
    )
    return reg


# ==============================================================================
# 4. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    # Path Data
    parser.add_argument('--train_csv', type=str, default='../output/train')
    parser.add_argument('--test_csv', type=str, default='../output/test')
    parser.add_argument('--img_dir', type=str, default='../output/filtered_images')
    # Output Results
    parser.add_argument('--output_dir', type=str, default='../output/analysis_results', help="Dossier racine des outputs")
    # Model config
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model_ep28.pt')
    parser.add_argument('--no_nn', action='store_true')
    parser.add_argument('--workers', type=int, default=max(1, multiprocessing.cpu_count()-1))
    parser.add_argument('--batch_size', type=int, default=16)
    
    # --- PARAMÈTRE PCA ---
    parser.add_argument('--pca_dim', type=int, default=32, help="Nombre de dimensions PCA pour les embeddings (Defaut: 50)")

    args = parser.parse_args()

    # Création dossier racine
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_nn = not args.no_nn

    print(f"[INIT] Mode: {'HYBRIDE (NN + XGB)' if use_nn else 'TABULAIRE (XGB Only)'}")

    # 1. SETUP DATA
    print("[INIT] Préparation des données...")
    cont_cols, cat_cols, _ = get_cols_config()
    scaler, medians, modes, cat_mappings, cat_dims = prepare_preprocessors(args.train_csv, cont_cols, cat_cols)
    tokenizer = AutoTokenizer.from_pretrained('almanach/camembert-base')

    train_ds = RealEstateDataset(args.train_csv, args.img_dir, tokenizer, scaler, medians, modes, cat_mappings, cont_cols, cat_cols)
    test_ds = RealEstateDataset(args.test_csv, args.img_dir, tokenizer, scaler, medians, modes, cat_mappings, cont_cols, cat_cols)

    # Patch pour désactiver le chargement d'image si on est en mode "no_nn"
    if not use_nn:
        print("[OPTIM] Mode Tabulaire : Images désactivées.")
        def fast_load_images(self, pid): return torch.zeros(1, 3, 224, 224), False
        train_ds.load_images = types.MethodType(fast_load_images, train_ds)
        test_ds.load_images = types.MethodType(fast_load_images, test_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=real_estate_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=real_estate_collate_fn)

    # 2. SETUP MODEL NN
    model = None
    embedding_dim = 0
    if use_nn:
        print(f"[MODEL] Chargement NN: {args.model_path}")
        # Chargement structure
        model = SOTARealEstateModel(len(cont_cols), cat_dims, 'convnext_large.fb_in1k', 'almanach/camembert-base', 512, 4, True).to(device)

        # Chargement poids
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state.items()})

        # [CHIRURGIE] On remplace la tête de prédiction par une identité
        # Le modèle ne sortira plus un prix, mais le vecteur de 512 features
        model.head_price = nn.Identity() 
        embedding_dim = 512

    # 3. EXTRACTION
    print("\n--- PHASE 1 : EXTRACTION ---")
    X_train, y_train, m_train = extract_dataset_features(train_loader, model, device, use_nn)
    X_test, y_test, m_test = extract_dataset_features(test_loader, model, device, use_nn)

    # Noms des features (Base Tabulaire)
    feature_names = cont_cols + cat_cols
    nb_tab_features = len(feature_names)

    # ==========================================================================
    # 3.5. RÉDUCTION DE DIMENSION (PCA) - SEULEMENT SI NN ACTIVÉ
    # ==========================================================================
    if use_nn:
        print(f"\n[PCA] Compression des embeddings (512 -> {args.pca_dim})...")
        print("      Objectif : Équilibrer l'importance Tabulaire vs Deep Learning.")

        # Séparation : Tabulaire (gauche) vs Embeddings (droite)
        X_train_tab = X_train[:, :nb_tab_features]
        X_train_emb = X_train[:, nb_tab_features:]
        
        X_test_tab = X_test[:, :nb_tab_features]
        X_test_emb = X_test[:, nb_tab_features:]

        # Entraînement PCA sur TRAIN uniquement (pour ne pas tricher)
        pca = PCA(n_components=args.pca_dim, random_state=42)
        X_train_emb_pca = pca.fit_transform(X_train_emb)
        X_test_emb_pca = pca.transform(X_test_emb) # Applique la transfo du train au test

        # Analyse de la variance conservée
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"      -> Variance conservée : {explained_var:.2%}")

        # Reconstitution : Tabulaire + PCA Embeddings
        X_train = np.hstack([X_train_tab, X_train_emb_pca])
        X_test = np.hstack([X_test_tab, X_test_emb_pca])

        # Mise à jour des noms de features pour les plots d'importance
        feature_names += [f"PCA_Emb_{i}" for i in range(args.pca_dim)]
        
        # Nettoyage mémoire des grosses matrices inutiles
        del X_train_tab, X_train_emb, X_test_tab, X_test_emb
        gc.collect()

    elif not use_nn:
        # Si pas de NN, pas de PCA, on reste sur les features tabulaires
        print("\n[PCA] Pas de Neural Network -> Pas de PCA nécessaire.")

    # 4. TRAINING & EVALUATION & SAVING
    print("\n--- PHASE 2 : XGBOOST & GENERATION RAPPORTS ---")

    # === VENTE ===
    mask_tr_v = m_train[:, 0] == 1
    mask_te_v = m_test[:, 0] == 1
    
    if np.sum(mask_tr_v) > 0:
        xgb_vente = train_xgb_core(
            X_train[mask_tr_v], y_train[mask_tr_v, 0],
            X_test[mask_te_v], y_test[mask_te_v, 0],
            label="VENTE"
        )
        if xgb_vente:
            model_path = os.path.join(args.output_dir, "xgb_vente.json")
            xgb_vente.save_model(model_path)
            evaluate_and_save(
                xgb_vente, 
                X_test[mask_te_v], 
                y_test[mask_te_v, 0], 
                feature_names, 
                "VENTE",
                args.output_dir
            )

    # === LOCATION ===
    mask_tr_l = m_train[:, 1] == 1
    mask_te_l = m_test[:, 1] == 1

    if np.sum(mask_tr_l) > 0:
        xgb_loc = train_xgb_core(
            X_train[mask_tr_l], y_train[mask_tr_l, 1],
            X_test[mask_te_l], y_test[mask_te_l, 1],
            label="LOCATION"
        )
        if xgb_loc:
            model_path = os.path.join(args.output_dir, "xgb_location.json")
            xgb_loc.save_model(model_path)
            evaluate_and_save(
                xgb_loc, 
                X_test[mask_te_l], 
                y_test[mask_te_l, 1], 
                feature_names, 
                "LOCATION",
                args.output_dir
            )

    print(f"\n Analyse terminée. Résultats disponibles dans : {args.output_dir}")

if __name__ == "__main__":
    main()
