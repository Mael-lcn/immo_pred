import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
# 1. VISUALISATION (Inchangée mais utilise les inputs corrigés)
# ==============================================================================
def save_full_analysis(images, input_ids, 
                       attn_img, attn_txt, 
                       feature_grads, feature_names,
                       pred_price, real_price, 
                       batch_idx, sample_idx, save_dir, tokenizer):
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    num_imgs = images.shape[0]

    fig = plt.figure(figsize=(20, 12))
    gs_top = gridspec.GridSpec(1, num_imgs + 1, width_ratios=[1]*num_imgs + [2])
    gs_bottom = gridspec.GridSpec(1, 1)
    gs_top.update(bottom=0.45, top=0.95)     
    gs_bottom.update(bottom=0.05, top=0.35)  

    # A. Images
    # Normalisation attention pour visu
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
        
        importance = norm_scores[i].item()
        color = plt.cm.coolwarm(importance)
        linewidth = 2 + (importance * 5)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(linewidth)
        ax.set_title(f"Img {i+1}\nAttn: {attn_img[i].item():.2f}", color=color, fontweight='bold', fontsize=10)
        ax.axis('off')

    # B. Texte
    ax_txt = plt.subplot(gs_top[-1])
    ax_txt.axis('off')
    full_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    display_text = (full_text[:400] + '...') if len(full_text) > 400 else full_text
    
    # Calcul erreur visuelle
    error_pct = abs(pred_price - real_price) / (real_price + 1) * 100
    color_res = 'green' if error_pct < 10 else 'red'

    info_str = (
        f"PRIX PREDIT : {pred_price:,.0f} €\n"
        f"PRIX REEL   : {real_price:,.0f} €\n"
        f"ERREUR      : {error_pct:.2f} %\n\n"
        f"SCORES ATTENTION :\n"
        f" - Images (Total) : {attn_img.sum():.3f}\n"
        f" - Texte          : {attn_txt:.3f}\n"
        f"-----------------\n"
        f"{display_text}"
    )
    ax_txt.text(0.05, 0.95, info_str, va='top', ha='left', fontsize=11, fontfamily='monospace',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor=color_res, boxstyle='round,pad=1'), transform=ax_txt.transAxes)

    # C. Features Tabulaires
    ax_feat = plt.subplot(gs_bottom[0])
    indices = np.argsort(feature_grads)[::-1] 
    sorted_grads = np.array(feature_grads)[indices]
    sorted_names = np.array(feature_names)[indices]
    
    top_n = 20
    sns.barplot(x=sorted_names[:top_n], y=sorted_grads[:top_n], ax=ax_feat, palette="viridis", hue=sorted_names[:top_n], legend=False)
    ax_feat.set_title("IMPORTANCE DES CARACTÉRISTIQUES (Gradients sur Log-Price)", fontsize=14, fontweight='bold')
    ax_feat.tick_params(axis='x', rotation=45, labelsize=9)
    ax_feat.grid(axis='y', linestyle='--', alpha=0.5)

    filename = f"analyse_batch{batch_idx}_sample{sample_idx}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=100, bbox_inches='tight')
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

    print(f"[INFO] Analyse visuelle sur les {max_visu_batches} premiers batchs...")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluation")):
        
        # --- A. Chargement GPU ---
        # Utilisation des clés du dictionnaire real_estate_collate_fn
        imgs = batch['images'].to(device, non_blocking=True)
        img_masks = batch['image_masks'].to(device, non_blocking=True)
        
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        text_mask = batch['text_mask'].to(device, non_blocking=True)
        
        x_cat = batch['x_cat'].to(device, non_blocking=True)
        x_cont = batch['x_cont'].to(device, non_blocking=True).clone().detach()
        x_cont.requires_grad = True # CRUCIAL pour Feature Importance
        
        targets = batch['targets'].to(device, non_blocking=True)
        masks = batch['masks'].to(device, non_blocking=True)
        
        # --- B. Forward (Avec Attention) ---
        # On active les gradients uniquement pour l'explicabilité, pas pour update les poids
        with torch.set_grad_enabled(True):
            p_vente_log, p_loc_log, attentions = model(
                images=imgs, 
                image_masks=img_masks, 
                input_ids=input_ids, 
                text_mask=text_mask, 
                x_cont=x_cont, 
                x_cat=x_cat, 
                return_attn=True
            )
            
            # --- C. Feature Importance (Calcul des gradients sur la sortie) ---
            gradients = None
            if batch_idx < max_visu_batches:
                # On choisit la sortie pertinente selon si c'est Vente ou Loc
                score_to_explain = (p_vente_log * masks[:, 0].unsqueeze(1)).sum() + (p_loc_log * masks[:, 1].unsqueeze(1)).sum()
                score_to_explain.backward(retain_graph=True)
                gradients = x_cont.grad.abs().cpu().numpy()
                model.zero_grad() # Reset des gradients stockés

        # --- D. Visualisation ---
        if batch_idx < max_visu_batches and gradients is not None:
            with torch.no_grad(): 
                # On prend la dernière couche d'attention
                last_attn = attentions[-1] 
                # Moyenne sur les têtes d'attention (B, Heads, Q, K) -> (B, Q, K)
                if last_attn.dim() == 4: last_attn = last_attn.mean(dim=1)

                # Conversion Log -> Exp pour l'affichage humain
                p_v_euro = torch.exp(p_vente_log).detach().cpu().numpy().flatten()
                p_l_euro = torch.exp(p_loc_log).detach().cpu().numpy().flatten()
                
                t_v_euro = torch.exp(targets[:, 0]).detach().cpu().numpy().flatten()
                t_l_euro = torch.exp(targets[:, 1]).detach().cpu().numpy().flatten()

                for i in range(imgs.shape[0]):
                    is_vente = masks[i, 0] > 0
                    is_loc = masks[i, 1] > 0
                    if not (is_vente or is_loc): continue

                    # Attention du Token CLS (Tabulaire) vers Image/Textes
                    cls_attn = last_attn[i, -1, :] 
                    attn_imgs = cls_attn[:-1] # Les N premiers sont les images
                    attn_txt = cls_attn[-1]   # Le dernier est le texte
                    feat_grads = gradients[i]

                    real_p = t_v_euro[i] if is_vente else t_l_euro[i]
                    pred_p = p_v_euro[i] if is_vente else p_l_euro[i]

                    save_full_analysis(
                        imgs[i], input_ids[i], attn_imgs, attn_txt, feat_grads, feature_names,
                        pred_p, real_p, batch_idx, i, attn_dir, tokenizer
                    )

        # --- E. Stockage Métriques (En Euros) ---
        with torch.no_grad():
            mask_vente = masks[:, 0].cpu().numpy().astype(bool)
            mask_loc = masks[:, 1].cpu().numpy().astype(bool)
            
            # Conversion Log -> Exp immédiate pour les métriques
            curr_p_vente = torch.exp(p_vente_log).detach().cpu().numpy().flatten()
            curr_p_loc = torch.exp(p_loc_log).detach().cpu().numpy().flatten()
            
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
    # Filtrage sécurité
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
    print(f"    MSE : {mse:.2e}")

    plt.figure(figsize=(10, 10))
    plt.scatter(t_clean, p_clean, alpha=0.4, s=10, c='#1f77b4')
    
    # Ligne parfaite x=y
    lims = [np.min([t_clean.min(), p_clean.min()]), np.max([t_clean.max(), p_clean.max()])]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    
    title_str = (f"Performance {cat_name}\n"
                 f"$R^2$ = {r2:.3f}  |  MAE = {mae:,.0f} €")
    plt.title(title_str, fontsize=14, fontweight='bold')
    plt.xlabel("Prix Réel (€)")
    plt.ylabel("Prix Estimé (€)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, f"global_metrics_{cat_name}.png"))
    plt.close()


# ==============================================================================
# 4. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', default='../output/train') 
    parser.add_argument('--test_csv', default='../output/test')
    parser.add_argument('--img_dir', default='../output/filtered_images')
    parser.add_argument('--model_path', default='checkpoints/model_ep30.pt') 
    parser.add_argument('--output_dir', default='../output/evaluation_results_final')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"[INIT] Device: {device}")

    # 1. Calibration (SUR LE TRAIN SET pour être cohérent)
    print("[INIT] Calibration Préprocesseurs (sur Train)...")
    num_cols, cat_cols, text_cols = get_cols_config()
    
    # IMPERATIF: On utilise le train set pour fitter le scaler/modes/mapping
    scaler, medians, modes, cat_mappings, cat_dims = prepare_preprocessors(args.train_csv, num_cols, cat_cols)

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('almanach/camembert-base')

    # 3. Dataset (Test)
    print(f"[DATA] Chargement Test Set: {args.test_csv}")
    ds = RealEstateDataset(
        df_or_folder=args.test_csv, 
        img_dir=args.img_dir, 
        tokenizer=tokenizer, 
        scaler=scaler, 
        medians=medians, 
        modes=modes,          # <--- AJOUT CRUCIAL
        cat_mappings=cat_mappings, 
        cont_cols=num_cols, 
        cat_cols=cat_cols
    )
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=real_estate_collate_fn)

    # 4. Modèle
    print(f"[MODEL] Chargement Architecture SOTA (Cont:{len(num_cols)}, Cat:{len(cat_dims)})...")
    model = SOTARealEstateModel(
        num_continuous=len(num_cols), 
        cat_cardinalities=cat_dims, 
        img_model_name='convnext_large.fb_in1k', 
        text_model_name='almanach/camembert-base',
        fusion_dim=512, depth=4, freeze_encoders=True
    ).to(device)

    # 5. Chargement Poids
    if os.path.exists(args.model_path):
        print(f"[MODEL] Chargement des poids depuis {args.model_path}")
        state_dict = torch.load(args.model_path, map_location=device)
        
        # Nettoyage préfixe torch.compile
        new_state_dict = {}
        for k, v in state_dict.items():
            key = k[10:] if k.startswith("_orig_mod.") else k
            new_state_dict[key] = v
        
        model.load_state_dict(new_state_dict)
    else:
        print(f"ERREUR: Checkpoint introuvable: {args.model_path}")
        exit(1)

    # 6. Exécution
    (pv, tv), (pl, tl) = run_eval_and_explain(model, loader, device, tokenizer, args.output_dir, num_cols)

    # 7. Sauvegarde
    print("[RESULT] Génération des graphiques globaux...")
    save_metrics(pv, tv, "VENTE", args.output_dir)
    save_metrics(pl, tl, "LOCATION", args.output_dir)

    print(f"[FIN] Terminé. Voir dossier: {args.output_dir}")


if __name__ == "__main__":
    main()
