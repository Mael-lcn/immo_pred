import os
import argparse
import json
import shutil

from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F

import open_clip
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from transformers import AutoImageProcessor, AutoModel, BitsAndBytesConfig




def list_images(folder):
    """
    Retourne la liste triée des chemins d'images dans `folder`.
    Extensions reconnues : jpg, jpeg, png, bmp, webp, tiff.
    """
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))
    return sorted(paths)



def load_clip_filter(device,
                     model_name='ViT-B-32',
                     pretrained='openai',
                     threshold_indoor=0.22,
                     threshold_bad=0.25,
                     threshold_outdoor=0.25,
                     debug=False):
    """
    Charge open_clip et renvoie les embeddings textes (outdoor / indoor / bad).
    """

    print(f"[load_clip_filter] device={device}, model={model_name}, pretrained={pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    # Outdoor
    outdoor_texts = [
        "an outdoor scene", "a garden", "a backyard", "a street",
        "a city street", "a building facade", "an exterior of a house", "an exterior",
        "an empty lot", "a construction site", "a building plot", "a garden area",
        "a yard under construction", "an open land", "a terrain", "a backyard garden",
        "a landscaped garden", "a fenced land", "a vacant land"
    ]
    # Indoor
    indoor_texts = [
        "an indoor scene", "an interior", "a living room", "a bedroom", "a kitchen",
        "an apartment interior", "a bathroom", "a hallway", "a dining room",
    ]
    # Junk (logo / plan / document)
    bad_texts = [
        "a logo", "company logo", "logo", "a floor plan", "a blueprint", "a map", "a watermark",
        "a document", "a scanned document", "a document photo", "a screenshot", "a business card",
        "a sign"
    ]

    with torch.no_grad():
        tokens_out = tokenizer(outdoor_texts).to(device)
        tokens_in = tokenizer(indoor_texts).to(device)
        tokens_bad = tokenizer(bad_texts).to(device)

        emb_out = model.encode_text(tokens_out).float()
        emb_in = model.encode_text(tokens_in).float()
        emb_bad = model.encode_text(tokens_bad).float()

        emb_out /= emb_out.norm(dim=-1, keepdim=True)
        emb_in /= emb_in.norm(dim=-1, keepdim=True)
        emb_bad /= emb_bad.norm(dim=-1, keepdim=True)

    return {
        "model": model,
        "preprocess": preprocess,
        "tokenizer": tokenizer,
        "emb_outdoor": emb_out,
        "emb_indoor": emb_in,
        "emb_bad": emb_bad,
        "threshold_indoor": threshold_indoor,
        "threshold_bad": threshold_bad,
        "threshold_outdoor": threshold_outdoor,
        "debug": debug,
        "device": device
    }


def is_desired_image_clip(filter_obj, pil_image,
                          margin_in=0.08, margin_out=0.06):
    """
    Décision améliorée :
      - reject si bad dominant (max_bad > max_in + margin_in OR max_bad > threshold_bad)
      - reject si outdoor dominant (max_out > max_in + margin_out OR max_out > threshold_outdoor)
      - keep si indoor suffisamment fort (max_in > threshold_indoor OR max_in - max_out > margin_in)
      - else reject
    Retourne (keep: bool, sims: dict)
    """
    model = filter_obj["model"]
    preprocess = filter_obj["preprocess"]
    device = filter_obj["device"]
    emb_in = filter_obj["emb_indoor"]
    emb_out = filter_obj["emb_outdoor"]
    emb_bad = filter_obj["emb_bad"]
    t_in = filter_obj["threshold_indoor"]
    t_bad = filter_obj["threshold_bad"]
    t_out = filter_obj.get("threshold_outdoor", 0.25)
    debug = filter_obj["debug"]

    x = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = model.encode_image(x).float()
        img_emb /= img_emb.norm(dim=-1, keepdim=True)

        sim_in = (img_emb @ emb_in.T).cpu().numpy().squeeze()
        sim_out = (img_emb @ emb_out.T).cpu().numpy().squeeze()
        sim_bad = (img_emb @ emb_bad.T).cpu().numpy().squeeze()

    max_in = float(np.max(sim_in)) if getattr(sim_in, 'size', 1) > 0 else 0.0
    max_out = float(np.max(sim_out)) if getattr(sim_out, 'size', 1) > 0 else 0.0
    max_bad = float(np.max(sim_bad)) if getattr(sim_bad, 'size', 1) > 0 else 0.0

    # 1) reject if clearly "bad"
    if (max_bad > max_in + margin_in) or (max_bad > t_bad):
        keep = False
        reason = "bad"
    # 2) reject if clearly outdoor
    elif (max_out > max_in + margin_out) or (max_out > t_out):
        keep = False
        reason = "outdoor"
    # 3) keep if clearly indoor
    elif (max_in > t_in) or (max_in - max_out > margin_in):
        keep = True
        reason = "indoor"
    else:
        keep = False
        reason = "uncertain"

    sims = {"indoor": max_in, "outdoor": max_out, "bad": max_bad, "reason": reason}
    if debug:
        print(f"[is_desired_image_clip] sims={sims} -> keep={keep}")
    return keep, sims



def load_visual_model():
    """
    Charge DINOv3 en 8-bit via HF + bitsandbytes
    Retourne : model, processor
    """

    model_id = 'facebook/dinov3-vits16-pretrain-lvd1689m'

    quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, quantization_config=quant_cfg, device_map="auto")
    print("[load_visual_model] HF DINOv3 chargé en 8-bit (bitsandbytes).")

    model.eval()
    return model, processor


def embed_image(model, processor, pil_image, device='cuda'):
    """
    Calcule et renvoie l'embedding d'une image PIL sous forme de vecteur numpy normalisé.
    Comporte deux chemins clairs :
      On utilise le pipeline HF (pixel_values -> model(**inputs))
    """

    device_t = torch.device(device)

    # traitement HF (AutoModel) : construire dict d'inputs et envoyer chaque tensor sur device
    inputs = processor(images=pil_image, return_tensors="pt")
    # déplacer chaque tensor vers device
    inputs = {k: v.to(device_t) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)

    # extraire features : prefer pooler_output puis last_hidden_state.mean
    feats = out.pooler_output

    # garantir forme (B, D) puis normaliser
    if feats.dim() > 2:
        feats = feats.flatten(1)
    feats = F.normalize(feats.float(), dim=-1)

    return feats.cpu().numpy().squeeze()


def run_pipeline(args, n_final=5, max_pca_components=16, clip_debug=False):
    """
    Exécute le pipeline complet :
      1) Liste les images
      2) Filtre via CLIP (garde seulement les vraies photos de pièces)
      3) Embeddings visuels
      4) PCA
      5) KMeans + sélection d'un exemplaire par cluster
    """
    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = list_images(args.input_dir)
    print(f"Found {len(image_paths)} images")

    device = 'cuda'

    # 1) CLIP filter
    clip_filter = load_clip_filter(device=device,
                                   model_name=args.clip_model,
                                   pretrained=args.clip_pretrained,
                                   threshold_indoor=0.20,
                                   threshold_bad=0.30,
                                   debug=clip_debug)

    # 2) filtrage images (on garde uniquement les vraies photos de pièces)
    kept = []
    for p in image_paths:
        # lecture d'image (gestion d'erreurs de fichiers)
        try:
            img = Image.open(p).convert('RGB')
        except Exception as e:
            print(f"[run_pipeline] skip (cannot open): {p} -> {e}")
            continue

        keep, sims = is_desired_image_clip(clip_filter, img)
        if keep:
            kept.append(p)
        elif clip_debug:
            print(f"[run_pipeline] dropped {p} sims={sims}")

    print(f"Kept {len(kept)} images after CLIP filtering")
    if not kept:
        print("No images left after filter. Exiting.")
        return

    # 3) charger modèle visuel et faire embeddings
    model, processor = load_visual_model()

    embs = []
    kept_imgs = []
    for p in kept:
        try:
            img = Image.open(p).convert('RGB')
        except Exception as e:
            print(f"[run_pipeline] can't open for embed: {p} -> {e}")
            continue
        vec = embed_image(model, processor, img, device=device)
        embs.append(vec)
        kept_imgs.append(p)

    if len(embs) == 0:
        print("No embeddings produced. Exiting.")
        return

    embs = np.vstack(embs)
    print(f"Computed embeddings for {len(embs)} images")

    # 4) PCA (nombre de composantes borné par n_samples et n_features)
    n_samples, n_features = embs.shape[0], embs.shape[1]
    n_components_pca = min(max_pca_components, n_features, n_samples)
    n_components_pca = max(1, n_components_pca)
    pca = PCA(n_components=n_components_pca)
    embs_pca = pca.fit_transform(embs)
    print(f"PCA done (n_components={n_components_pca}), retained var={pca.explained_variance_ratio_.sum():.3f}")

    # 5) KMeans : une image représentative par cluster
    n_clusters = min(n_final, len(kept_imgs))
    if n_clusters <= 0:
        print("n_final invalid (<=0). Exiting.")
        return

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(embs_pca)

    selected = []
    for lab in range(n_clusters):
        idxs = np.where(labels == lab)[0]
        if idxs.size == 0:
            continue
        centroid = embs_pca[idxs].mean(axis=0)
        dists = np.linalg.norm(embs_pca[idxs] - centroid, axis=1)
        chosen = idxs[np.argmin(dists)]
        selected.append((kept_imgs[chosen], lab))

    # écriture des images sélectionnées
    sel_dir = os.path.join(args.output_dir, 'selected')
    os.makedirs(sel_dir, exist_ok=True)
    for p, lab in selected:
        dst = os.path.join(sel_dir, f'cluster_{lab}_{os.path.basename(p)}')
        shutil.copyfile(p, dst)

    print(f"Selected {len(selected)} representative images")
    for p, lab in selected:
        print(f" - cluster {lab}: {p}")

    # résumé
    meta = {
        "total": len(image_paths),
        "kept": len(kept),
        "selected": len(selected),
        "clusters": len(set(l for _, l in selected))
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='../output/test_img/2507940269')
    parser.add_argument('--output_dir', default='../output/cluster_res/')
    parser.add_argument('--clip_model', default='ViT-B-32')
    parser.add_argument('--clip_pretrained', default='openai')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--n_final', type=int, default=5,
                        help="Nombre final d'images à garder (ex: 1 par pièce).")
    parser.add_argument('--clip_debug', action='store_true', help="Affiche les similitudes CLIP pour debug")
    args = parser.parse_args()

    print('\n=== ENV INFO ===')
    print('torch version:', torch.__version__)
    print('cuda available:', torch.cuda.is_available())
    print('bf16 supported:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
    print('=== END ENV INFO ===\n')

    run_pipeline(args, n_final=args.n_final, max_pca_components=16, clip_debug=args.clip_debug)


if __name__ == '__main__':
    main()
