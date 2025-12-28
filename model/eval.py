import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.switch_backend('agg')

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
# 1. ANALYSE GLOBALE : GRADIENS (TABULAIRE) & ATTENTION (IMAGES vs TEXTE)
# ==============================================================================
def save_global_analysis(all_grads, all_attn_img, all_attn_txt, feature_names, errors, output_dir):
    """
    Génère 3 graphiques majeurs pour comprendre le comportement GLOBAL du modèle.
    1. Importance des Features (Gradients)
    2. Balance d'Attention (Images vs Texte)
    3. Distribution des Erreurs
    """
    print(f"[INFO] Génération des rapports globaux dans {output_dir}...")

    # --- A. FEATURES IMPORTANCE (Tabulaire) ---
    if all_grads:
        mean_grads = np.mean(np.vstack(all_grads), axis=0)
        df_feat = pd.DataFrame({
            'Feature': feature_names,
            'Importance': mean_grads
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(data=df_feat.head(20), x='Importance', y='Feature', palette="magma")
        plt.title("TOP 20 CRITÈRES D'ESTIMATION (Impact Global)", fontsize=14, fontweight='bold')
        plt.xlabel("Importance Moyenne (Gradient Absolu)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "GLOBAL_1_feature_importance.png"), dpi=150)
        plt.close()
        df_feat.to_csv(os.path.join(output_dir, "global_feature_importance.csv"), index=False)

    # --- B. BALANCE D'ATTENTION (Images vs Texte) ---
    if all_attn_img and all_attn_txt:
        # On crée un DataFrame pour Seaborn
        data_attn = pd.DataFrame({
            'Score': all_attn_img + all_attn_txt,
            'Source': ['Images'] * len(all_attn_img) + ['Texte'] * len(all_attn_txt)
        })

        plt.figure(figsize=(8, 6))
        # Violin plot montre la densité de la distribution
        sns.violinplot(data=data_attn, x='Source', y='Score', palette="viridis", split=True)
        
        # Calcul des moyennes pour affichage
        mean_img = np.mean(all_attn_img)
        mean_txt = np.mean(all_attn_txt)
        
        plt.title(f"COMPÉTITION VISUELLE : IMAGES ({mean_img:.2f}) vs TEXTE ({mean_txt:.2f})", fontsize=12, fontweight='bold')
        plt.ylabel("Poids d'Attention du Modèle")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "GLOBAL_2_attention_balance.png"), dpi=150)
        plt.close()

    # --- C. DISTRIBUTION DES ERREURS ---
    if len(errors) > 0:
        plt.figure(figsize=(10, 6))
        # On clip les erreurs extrêmes pour la lisibilité graphique (-100% à +100%)
        clean_errors = np.clip(errors, -100, 100)

        sns.histplot(clean_errors, bins=50, kde=True, color="steelblue")
        plt.axvline(0, color='red', linestyle='--', linewidth=2, label="Estimation Parfaite")

        mean_err = np.mean(errors)
        med_err = np.median(errors)

        plt.title(f"DISTRIBUTION DE L'ERREUR (Moy: {mean_err:.1f}% | Med: {med_err:.1f}%)", fontsize=12, fontweight='bold')
        plt.xlabel("Pourcentage d'erreur (Prédiction vs Réel)")
        plt.ylabel("Nombre de biens")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "GLOBAL_3_error_distribution.png"), dpi=150)
        plt.close()


# ==============================================================================
# 2. MOTEUR D'ANALYSE
# ==============================================================================
def run_eval_and_explain(model, dataloader, device, tokenizer, output_dir, feature_names, max_visu_batches=5):
    model.eval() 
    attn_dir = os.path.join(output_dir, "exemples_locaux")
    os.makedirs(attn_dir, exist_ok=True)

    preds_vente, targets_vente = [], []
    preds_loc, targets_loc = [], []
    
    # --- STOCKAGE GLOBAL ---
    all_feature_grads = [] # Pour Feature Importance
    all_attn_sum_img = []  # Somme attention sur les images par bien
    all_attn_sum_txt = []  # Somme attention sur le texte par bien
    all_pct_errors = []    # Erreurs en % pour histogramme

    print(f"[INFO] Analyse approfondie en cours...")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Audit Modèle")):
        
        # --- A. Chargement ---
        imgs = batch['images'].to(device, non_blocking=True)
        img_masks = batch['image_masks'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        text_mask = batch['text_mask'].to(device, non_blocking=True)
        x_cat = batch['x_cat'].to(device, non_blocking=True)
        
        # Gradient activé sur le tabulaire continu pour l'importance
        x_cont = batch['x_cont'].to(device, non_blocking=True).clone().detach()
        x_cont.requires_grad = True 
        
        targets = batch['targets'].to(device, non_blocking=True)
        masks = batch['masks'].to(device, non_blocking=True)
        
        # --- B. Forward ---
        with torch.set_grad_enabled(True):
            p_vente_log, p_loc_log, attentions = model(
                images=imgs, image_masks=img_masks, input_ids=input_ids, 
                text_mask=text_mask, x_cont=x_cont, x_cat=x_cat, return_attn=True
            )
            
            # --- C. Feature Importance (Gradient) ---
            # On le fait sur quelques batchs pour ne pas saturer la RAM, 
            # ou sur tout si on veut être précis (ici on limite aux max_visu pour la vitesse)
            if batch_idx < max_visu_batches * 2: 
                score = (p_vente_log * masks[:, 0].unsqueeze(1)).sum() + (p_loc_log * masks[:, 1].unsqueeze(1)).sum()
                score.backward(retain_graph=True)
                gradients = x_cont.grad.abs().cpu().numpy()
                all_feature_grads.append(gradients)
                model.zero_grad()
            else:
                gradients = None

        # --- D. Collecte Statistiques Globales ---
        with torch.no_grad():
            # 1. Attention Balance (Dernière couche)
            last_attn = attentions[-1] 
            if last_attn.dim() == 4: last_attn = last_attn.mean(dim=1) # (B, Q, K)

            # Le Token Query est le CLS Tabulaire (le dernier de Q)
            # Les Keys sont [Img1...ImgN, Texte]
            cls_attn = last_attn[:, -1, :] # (B, N_Img + 1)

            # On sépare Img et Texte
            attn_imgs_vals = cls_attn[:, :-1] # Tout sauf le dernier
            attn_txt_vals  = cls_attn[:, -1]  # Le dernier

            # On somme l'attention portée à toutes les images valides
            sum_imgs = (attn_imgs_vals * img_masks).sum(dim=1).cpu().numpy()
            sum_txt  = attn_txt_vals.cpu().numpy()

            all_attn_sum_img.extend(sum_imgs)
            all_attn_sum_txt.extend(sum_txt)

            # 2. Prédictions & Erreurs
            p_v_euro = torch.exp(p_vente_log).cpu().numpy().flatten()
            t_v_euro = torch.exp(targets[:, 0]).cpu().numpy().flatten()
            p_l_euro = torch.exp(p_loc_log).cpu().numpy().flatten()
            t_l_euro = torch.exp(targets[:, 1]).cpu().numpy().flatten()

            m_v = masks[:, 0].cpu().numpy().astype(bool)
            m_l = masks[:, 1].cpu().numpy().astype(bool)

            # Stockage Vente
            if m_v.any():
                preds_vente.extend(p_v_euro[m_v])
                targets_vente.extend(t_v_euro[m_v])
                # Calcul erreur % pour histogramme
                errs = (p_v_euro[m_v] - t_v_euro[m_v]) / (t_v_euro[m_v] + 1) * 100
                all_pct_errors.extend(errs)

            # Stockage Loc
            if m_l.any():
                preds_loc.extend(p_l_euro[m_l])
                targets_loc.extend(t_l_euro[m_l])
                errs = (p_l_euro[m_l] - t_l_euro[m_l]) / (t_l_euro[m_l] + 1) * 100
                all_pct_errors.extend(errs)

    # --- E. GÉNÉRATION DES RAPPORTS ---
    save_global_analysis(
        all_feature_grads, 
        all_attn_sum_img, 
        all_attn_sum_txt, 
        feature_names, 
        np.array(all_pct_errors), 
        output_dir
    )

    return (np.array(preds_vente), np.array(targets_vente)), \
           (np.array(preds_loc), np.array(targets_loc))


# ==============================================================================
# 3. METRIQUES CLASSIQUES
# ==============================================================================
def save_metrics(preds, targets, cat_name, output_dir):
    mask = np.isfinite(preds) & np.isfinite(targets)
    p_clean = preds[mask]
    t_clean = targets[mask]

    if len(p_clean) == 0: 
        print(f"Pas de données pour {cat_name}")
        return

    mae = mean_absolute_error(t_clean, p_clean)
    mse = mean_squared_error(t_clean, p_clean)
    r2 = r2_score(t_clean, p_clean)

    print(f"\n>>> RÉSULTATS {cat_name}")
    print(f"    R²  : {r2:.4f}")
    print(f"    MAE : {mae:,.0f} €")
    print(f"    MSE : {mse:,.0f} €")

    # Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(t_clean, p_clean, alpha=0.3, s=5, c='#1f77b4')
    lims = [min(t_clean.min(), p_clean.min()), max(t_clean.max(), p_clean.max())]
    plt.plot(lims, lims, 'r--', linewidth=2)
    plt.title(f"{cat_name}: R²={r2:.3f} | MAE={mae:,.0f}€ MSE={mse:.3f} |")
    plt.xlabel("Prix Réel"); plt.ylabel("Prix Estimé")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"metrics_{cat_name}.png"))
    plt.close()


# ==============================================================================
# 4. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', default='../output/train') 
    parser.add_argument('--test_csv', default='../output/test')
    parser.add_argument('--img_dir', default='../output/filtered_images')
    parser.add_argument('--model_path', default='checkpoints/best_model.pt') 
    parser.add_argument('--output_dir', default='../output/evaluation_results_final')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    if torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"[INIT] Device: {device}")

    # 1. Calibration
    print("[INIT] Calibration Préprocesseurs (sur Train)...")
    num_cols, cat_cols, text_cols = get_cols_config()
    scaler, medians, modes, cat_mappings, cat_dims = prepare_preprocessors(args.train_csv, num_cols, cat_cols)

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('almanach/camembert-base')

    # 3. Dataset Test
    print(f"[DATA] Chargement Test Set: {args.test_csv}")
    ds = RealEstateDataset(
        df_or_folder=args.test_csv, img_dir=args.img_dir, tokenizer=tokenizer, 
        scaler=scaler, medians=medians, modes=modes, cat_mappings=cat_mappings, 
        cont_cols=num_cols, cat_cols=cat_cols
    )
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=real_estate_collate_fn)

    # 4. Modèle
    print(f"[MODEL] Chargement Architecture...")
    model = SOTARealEstateModel(
        len(num_cols), cat_dims, 'convnext_large.fb_in1k', 'almanach/camembert-base', 
        fusion_dim=512, depth=4, freeze_encoders=True
    ).to(device)

    # 5. Poids
    if os.path.exists(args.model_path):
        print(f"[MODEL] Chargement {args.model_path}")
        state_dict = torch.load(args.model_path, map_location=device)
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        print(f"ERREUR: Introuvable {args.model_path}"); exit(1)

    # 6. Exécution
    (pv, tv), (pl, tl) = run_eval_and_explain(model, loader, device, tokenizer, args.output_dir, num_cols)

    # 7. Metrics
    save_metrics(pv, tv, "VENTE", args.output_dir)
    save_metrics(pl, tl, "LOCATION", args.output_dir)
    print(f"[FIN] Résultats disponibles dans : {args.output_dir}")

if __name__ == "__main__":
    main()
