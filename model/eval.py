"""
evaluate_with_attention.py - V2 : Images + Texte + TABULAIRE.

Nouveautés :
- Visualisation "Feature Importance" pour les données tabulaires (via Gradients).
- Visualisation "Attention" pour les Images et le Texte.
- Calcul des métriques globales.
"""

import os
import argparse
import multiprocessing
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Mode silencieux
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
# 1. VISUALISATION COMPLETE (Attention + Features)
# ==============================================================================
def save_full_analysis(images, input_ids, 
                       attn_img, attn_txt, 
                       feature_grads, feature_names,
                       pred_price, real_price, 
                       batch_idx, sample_idx, save_dir, tokenizer):
    """
    Génère une planche complète : Attention Images/Texte + Importance Tabulaire.
    """
    # Config Image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    num_imgs = images.shape[0]

    # Création de la figure (Plus grande pour tout contenir)
    fig = plt.figure(figsize=(20, 10))
    
    # Grille : Haut = Images/Texte | Bas = Features Tabulaires
    gs_top = gridspec.GridSpec(1, num_imgs + 1, width_ratios=[1]*num_imgs + [2])
    gs_bottom = gridspec.GridSpec(1, 1)
    
    # Positionnement
    gs_top.update(bottom=0.45, top=0.95)     # Partie haute
    gs_bottom.update(bottom=0.05, top=0.35)  # Partie basse

    # --- A. ATTENTION IMAGES (Partie Haute Gauche) ---
    if attn_img.max() > 0:
        norm_scores = (attn_img - attn_img.min()) / (attn_img.max() - attn_img.min() + 1e-6)
    else:
        norm_scores = attn_img

    for i in range(num_imgs):
        ax = plt.subplot(gs_top[i])
        img = images[i].cpu() * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        # Cadre couleur
        importance = norm_scores[i].item()
        color = plt.cm.coolwarm(importance)
        linewidth = 2 + (importance * 5)
        
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(linewidth)
        
        ax.set_title(f"Img {i+1}\nScore: {attn_img[i].item():.2f}", color=color, fontweight='bold', fontsize=9)
        ax.axis('off')

    # --- B. INFO TEXTE (Partie Haute Droite) ---
    ax_txt = plt.subplot(gs_top[-1])
    ax_txt.axis('off')
    full_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    display_text = (full_text[:300] + '...') if len(full_text) > 300 else full_text
    
    error_pct = abs(pred_price - real_price) / (real_price + 1) * 100
    color_res = 'green' if error_pct < 10 else 'red'

    info_str = (
        f"PRIX PREDIT : {pred_price:,.0f} €\n"
        f"PRIX REEL   : {real_price:,.0f} €\n"
        f"ERREUR      : {error_pct:.2f} %\n\n"
        f"IMPORTANCE GLOBALE :\n"
        f" - Images : {attn_img.sum():.3f}\n"
        f" - Texte  : {attn_txt:.3f}\n"
        f"-----------------\n"
        f"{display_text}"
    )
    ax_txt.text(0.0, 0.5, info_str, va='center', ha='left', fontsize=12, fontfamily='monospace',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor=color_res, boxstyle='round,pad=1'))

    # --- C. IMPORTANCE TABULAIRE (Partie Basse) ---
    ax_feat = plt.subplot(gs_bottom[0])
    
    # On trie les features par importance
    indices = np.argsort(feature_grads)[::-1] # Décroissant
    sorted_grads = np.array(feature_grads)[indices]
    sorted_names = np.array(feature_names)[indices]
    
    # Barplot
    sns.barplot(x=sorted_names, y=sorted_grads, ax=ax_feat, palette="viridis", hue=sorted_names, legend=False)
    ax_feat.set_title("IMPORTANCE DES CARACTÉRISTIQUES (Impact sur le prix)", fontsize=14, fontweight='bold')
    ax_feat.set_ylabel("Sensibilité (Gradient)", fontsize=10)
    ax_feat.tick_params(axis='x', rotation=45, labelsize=9)
    ax_feat.grid(axis='y', linestyle='--', alpha=0.5)

    # Sauvegarde
    filename = f"analyse_batch{batch_idx}_sample{sample_idx}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=100)
    plt.close()


# ==============================================================================
# 2. MOTEUR D'ANALYSE
# ==============================================================================
def run_eval_and_explain(model, dataloader, device, tokenizer, output_dir, feature_names, max_visu_batches=5):
    model.eval()
    attn_dir = os.path.join(output_dir, "attention_analysis")
    os.makedirs(attn_dir, exist_ok=True)

    preds_vente, targets_vente = [], []
    preds_loc, targets_loc = [], []

    print(f"[INFO] Analyse visuelle (Images + Tabulaire) sur les {max_visu_batches} premiers batchs...")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluation")):
        
        imgs = batch['images'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        mask = batch['attention_mask'].to(device, non_blocking=True)
        x_cat = batch['tab_cat'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        masks = batch['masks'].to(device, non_blocking=True)
        
        # --- ASTUCE POUR TABULAR IMPORTANCE ---
        x_cont = batch['tab_cont'].to(device, non_blocking=True).clone().detach()
        x_cont.requires_grad = True

        # --- FORWARD ---
        with torch.set_grad_enabled(True):
            p_vente, p_loc, attentions = model(imgs, input_ids, mask, x_cont, x_cat, return_attn=True)
            
            # --- CALCUL DU GRADIENT (Feature Importance) ---
            if batch_idx < max_visu_batches:
                score_to_explain = (p_vente * masks[:, 0].unsqueeze(1)).sum() + (p_loc * masks[:, 1].unsqueeze(1)).sum()
                score_to_explain.backward(retain_graph=True)
                gradients = x_cont.grad.abs().cpu().numpy() 
                
                x_cont.requires_grad = False
                model.zero_grad()

        # --- VISUALISATION ---
        if batch_idx < max_visu_batches:
            with torch.no_grad(): 
                # === CORRECTION ICI ===
                last_attn = attentions[-1] 
                if last_attn.dim() == 4: # Si (Batch, Heads, Q, K)
                    last_attn = last_attn.mean(dim=1) 
                # Maintenant last_attn est (Batch, Q, K)
                # =======================

                p_v_np = p_vente.detach().cpu().numpy().flatten()
                p_l_np = p_loc.detach().cpu().numpy().flatten()
                t_v_np = torch.exp(targets[:, 0]).detach().cpu().numpy().flatten()
                t_l_np = torch.exp(targets[:, 1]).detach().cpu().numpy().flatten()

                for i in range(imgs.shape[0]):
                    is_vente = masks[i, 0] > 0
                    is_loc = masks[i, 1] > 0
                    if not (is_vente or is_loc): continue

                    # 1. Attention Images/Texte (Cross-Attention)
                    # last_attn est [B, Q, K]
                    # On prend l'index i du batch, le dernier élément Q, et tout K
                    cls_attn = last_attn[i, -1, :] 
                    
                    attn_imgs = cls_attn[:-1]
                    attn_txt = cls_attn[-1]

                    # 2. Importance Tabulaire (Gradients)
                    feat_grads = gradients[i]

                    # Prix
                    real_p = t_v_np[i] if is_vente else t_l_np[i]
                    pred_p = p_v_np[i] if is_vente else p_l_np[i]

                    # Génération Image
                    save_full_analysis(
                        imgs[i], input_ids[i], 
                        attn_imgs, attn_txt, 
                        feat_grads, feature_names,
                        pred_p, real_p, 
                        batch_idx, i, attn_dir, tokenizer
                    )

        # --- STOCKAGE MÉTRIQUES ---
        with torch.no_grad():
            mask_vente = masks[:, 0].cpu().numpy().astype(bool)
            mask_loc = masks[:, 1].cpu().numpy().astype(bool)
            
            curr_p_vente = p_vente.detach().cpu().numpy().flatten()
            curr_p_loc = p_loc.detach().cpu().numpy().flatten()
            curr_t_vente = torch.exp(targets[:, 0]).cpu().numpy().flatten()
            curr_t_loc = torch.exp(targets[:, 1]).cpu().numpy().flatten()

            if mask_vente.any():
                preds_vente.extend(curr_p_vente[mask_vente])
                targets_vente.extend(curr_t_vente[mask_vente])
            if mask_loc.any():
                preds_loc.extend(curr_p_loc[mask_loc])
                targets_loc.extend(curr_t_loc[mask_loc])

    return (np.array(preds_vente), np.array(targets_vente)), \
           (np.array(preds_loc), np.array(targets_loc))


# ==============================================================================
# 3. METRIQUES
# ==============================================================================
def save_metrics(preds, targets, cat_name, output_dir):
    mask = np.isfinite(preds) & np.isfinite(targets)
    p_clean = preds[mask]
    t_clean = targets[mask]

    if len(p_clean) == 0: return

    mae = mean_absolute_error(t_clean, p_clean)
    r2 = r2_score(t_clean, p_clean)

    print(f"\n>>> RÉSULTATS {cat_name}")
    print(f"    R² : {r2:.4f}")
    print(f"    MAE: {mae:,.0f} €")

    plt.figure(figsize=(8,8))
    plt.scatter(t_clean, p_clean, alpha=0.3, s=5)
    plt.plot([t_clean.min(), t_clean.max()], [t_clean.min(), t_clean.max()], 'r--')
    plt.title(f"{cat_name}: R²={r2:.3f}")
    plt.xlabel("Prix Réel")
    plt.ylabel("Prix Estimé")
    plt.savefig(os.path.join(output_dir, f"global_plot_{cat_name}.png"))
    plt.close()


# ==============================================================================
# 4. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', default='../output/train')
    parser.add_argument('--test_csv', default='../output/test')
    parser.add_argument('--img_dir', default='../output/filtered_images')
    parser.add_argument('--model_path', default='checkpoints_sota/model_ep20.pt')
    parser.add_argument('--output_dir', default='../output/evaluation_results2')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")

    # Calibration
    print("[INIT] Calibration...")
    num_cols, cat_cols, text_cols = get_cols_config()
    scaler, medians, cat_mappings, cat_dims = prepare_preprocessors(args.train_csv, num_cols, cat_cols)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('almanach/camembert-base')

    # Dataset Test
    ds = RealEstateDataset(args.test_csv, args.img_dir, tokenizer, scaler, medians, cat_mappings, num_cols, cat_cols)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=real_estate_collate_fn)

    # Modèle
    print("[INIT] Chargement Modèle...")
    model = SOTARealEstateModel(
        len(num_cols), cat_dims, 
        img_model_name='convnext_large.fb_in1k', 
        text_model_name='almanach/camembert-base',
        fusion_dim=512, depth=4, freeze_encoders=True
    ).to(device)

    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("ERREUR: Modèle introuvable.")
        return

    # Exécution avec noms des features pour l'affichage
    (pv, tv), (pl, tl) = run_eval_and_explain(model, loader, device, tokenizer, args.output_dir, num_cols)

    # Sauvegarde Metrics
    save_metrics(pv, tv, "VENTE", args.output_dir)
    save_metrics(pl, tl, "LOCATION", args.output_dir)

    print(f"[FIN] Résultats sauvegardés dans {args.output_dir}")


if __name__ == "__main__":
    main()
