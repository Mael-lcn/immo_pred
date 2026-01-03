"""
ai_part.py

Module contenant toute la logique d'intelligence artificielle :
1. Initialisation des modèles (CLIP, DINO) avec optimisations (Quantization 4-bit, Compilation).
2. Filtrage sémantique via CLIP (exclure plans, logos, extérieur si non désiré).
3. Clustering visuel via DINO + Agglomerative Clustering (sélection de diversité).
"""

import os
import time
from io import BytesIO
from typing import List, Tuple, Dict

from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel, BitsAndBytesConfig

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances



# ==============================================================================
# 1. CONFIGURATION & OPTIMISATIONS (BitsAndBytes / Torch Compile)
# ==============================================================================

def _make_bnb_config():
    """Tente de créer une config de quantization 4-bit NF4 pour économiser la VRAM."""
    try:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    except Exception:
        return None

def try_compile_model(model, name):
    """Tente de compiler le modèle avec torch.compile (Linux/WSL uniquement, gain de vitesse)."""
    compiled = False
    if hasattr(torch, "compile"):
        try:
            print(f"[AI] Tentative torch.compile() pour {name} ...")
            model = torch.compile(model)
            compiled = True
            print(f"[AI] Succès torch.compile() pour {name}")
        except Exception as e:
            print(f"[AI] Échec torch.compile() pour {name}: {e}")
    return model, compiled


# ==============================================================================
# 2. INITIALISATION DES MODÈLES
# ==============================================================================

def init_models(device="cuda",
                clip_hf_id="openai/clip-vit-base-patch32",
                visual_id="facebook/dinov2-base"):
    """
    Charge CLIP et le modèle Visuel (DINO).
    Retourne un tuple contenant les objets nécessaires à l'inférence.
    """
    # Détection automatique du device si 'auto' ou incorrect
    if device not in ["cpu", "cuda"]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[AI] Initialisation sur : {device}")

    # Configuration Quantization
    bnb_cfg = _make_bnb_config()
    if bnb_cfg:
        print("[AI] Quantization 4-bit activée.")

    # ---------------------------------------------------------
    # A. CHARGEMENT CLIP (Filtrage sémantique)
    # ---------------------------------------------------------
    print(f"[AI] Chargement CLIP : {clip_hf_id}...")
    try:
        proc_clip = CLIPProcessor.from_pretrained(clip_hf_id)

        # Mappage automatique pour quantization, sinon manuel
        clip_args = {"quantization_config": bnb_cfg, "device_map": "auto"} if bnb_cfg else {"device_map": device}
        if device == "cpu": clip_args = {} # Pas de device_map sur CPU

        model_clip = CLIPModel.from_pretrained(clip_hf_id, **clip_args)
        model_clip.eval()

        if device == "cpu": model_clip = model_clip.to("cpu")

        model_clip, _ = try_compile_model(model_clip, "CLIP")
    except Exception as e:
        print(f"[ERREUR] Impossible de charger CLIP : {e}")
        raise e

    # Pré-calcul des prototypes textuels (Outdoor vs Indoor vs Bad)
    # Cela évite de ré-encoder le texte à chaque image.
    device_t = model_clip.device

    outdoor_txt = ["an outdoor scene", "garden", "backyard", "street view", "house exterior", "building facade"]
    indoor_txt  = ["an indoor scene", "living room", "bedroom", "kitchen", "bathroom", "empty room"]
    bad_txt     = ["a floor plan", "a blueprint", "a map", "text document", "plot plan", "technical drawing", "logo"]

    with torch.no_grad():
        tok_out = proc_clip(text=outdoor_txt, return_tensors="pt", padding=True).to(device_t)
        tok_in  = proc_clip(text=indoor_txt, return_tensors="pt", padding=True).to(device_t)
        tok_bad = proc_clip(text=bad_txt, return_tensors="pt", padding=True).to(device_t)

        emb_out = F.normalize(model_clip.get_text_features(**tok_out).float(), dim=-1)
        emb_in  = F.normalize(model_clip.get_text_features(**tok_in).float(), dim=-1)
        emb_bad = F.normalize(model_clip.get_text_features(**tok_bad).float(), dim=-1)

    # Contexte CLIP à passer aux fonctions suivantes
    clip_context = {
        "model": model_clip,
        "processor": proc_clip,
        "emb_out": emb_out,
        "emb_in": emb_in,
        "emb_bad": emb_bad,
        # Seuils de tolérance (ajustables)
        "thresholds": {"bad": 0.26}, 
        "device": device_t
    }

    # ---------------------------------------------------------
    # B. CHARGEMENT DINO (Embeddings Visuels & Clustering)
    # ---------------------------------------------------------
    print(f"[AI] Chargement Visuel (DINO) : {visual_id}...")
    try:
        proc_vis = AutoImageProcessor.from_pretrained(visual_id)
        vis_args = {"quantization_config": bnb_cfg, "device_map": "auto"} if bnb_cfg else {"device_map": device}
        if device == "cpu": vis_args = {}

        model_vis = AutoModel.from_pretrained(visual_id, **vis_args)
        model_vis.eval()

        if device == "cpu": model_vis = model_vis.to("cpu")

        model_vis, _ = try_compile_model(model_vis, "DINO")
    except Exception as e:
        print(f"[ERREUR] Impossible de charger DINO : {e}")
        raise e

    return clip_context, model_vis, proc_vis


# ==============================================================================
# 3. FONCTIONS UTILITAIRES (Conversion, Batching)
# ==============================================================================

def _pil_from_bytes(b: bytes):
    """Convertit des bytes en PIL Image RGB."""
    return Image.open(BytesIO(b)).convert("RGB")


def embed_images_in_batches(model, processor, pil_images, batch_size=16):
    """
    Génère les embeddings DINO pour une liste d'images.
    Gère le batching pour ne pas saturer la VRAM.
    """
    device = model.device
    all_embs = []
    
    # Boucle par batch
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Extraction du vecteur caractéristique (CLS token ou Mean Pooling)
        # DINOv2 : last_hidden_state[:, 0, :] est souvent le CLS token
        if hasattr(outputs, "last_hidden_state"):
            # On prend le CLS token (index 0) pour DINOv2, c'est le standard pour la similarité globale
            feats = outputs.last_hidden_state[:, 0, :] 
        elif hasattr(outputs, "pooler_output"):
            feats = outputs.pooler_output
        else:
            # Fallback générique : moyenne
            feats = outputs[0].mean(dim=1)

        feats = F.normalize(feats.float(), dim=-1)
        all_embs.append(feats.cpu().numpy())

    if not all_embs:
        return np.zeros((0, 0))
        
    return np.vstack(all_embs)


# ==============================================================================
# 4. FILTRAGE CLIP
# ==============================================================================

def filter_with_clip(clip_ctx, pil_imgs, batch_size=16):
    """
    Passe les images dans CLIP et décide si on garde (Keep) ou jette (Drop).
    Critère : Si similitude avec "Bad" (plan, logo) > Seuil OU > (Indoor+Outdoor).
    """
    model = clip_ctx["model"]
    proc = clip_ctx["processor"]
    device = clip_ctx["device"]
    
    kept_indices = []
    
    for i in range(0, len(pil_imgs), batch_size):
        batch = pil_imgs[i:i+batch_size]
        # Preprocessing CLIP
        inputs = proc(images=batch, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            img_feats = model.get_image_features(**inputs)
            img_feats = F.normalize(img_feats.float(), dim=-1)
            
            # Similarité Cosine
            sim_in  = (img_feats @ clip_ctx["emb_in"].T)
            sim_out = (img_feats @ clip_ctx["emb_out"].T)
            sim_bad = (img_feats @ clip_ctx["emb_bad"].T)
  
            # Max sur chaque catégorie
            max_in  = sim_in.max(dim=1).values
            max_out = sim_out.max(dim=1).values
            max_bad = sim_bad.max(dim=1).values

            # Logique de décision vectorisée
            # On rejette si c'est "bad" (plan/logo)
            is_bad = (max_bad > clip_ctx["thresholds"]["bad"]) | (max_bad > torch.maximum(max_in, max_out))

            keep_mask = ~is_bad

            # Conversion CPU pour indexation
            keep_mask = keep_mask.cpu().numpy()

            # Récupération des index globaux
            current_indices = np.arange(i, i + len(batch))
            kept_indices.extend(current_indices[keep_mask])

    return kept_indices


# ==============================================================================
# 5. CLUSTERING INTELLIGENT
# ==============================================================================

def process_listing_smart_clustering(
    listing_id: str,
    images_bytes: List[Tuple[str, bytes]],
    results_dir: str,
    models_context: tuple,
    device: str = "cuda",
    similarity_threshold: float = 0.25,
    min_images: int = 3,
    max_images: int = 15
) -> Dict:
    """
    Pipeline complet SOTA :
    1. Décodage images.
    2. Filtrage CLIP (suppression plans, documents).
    3. Embedding DINO (représentation sémantique structurelle).
    4. Clustering Agglomératif (regroupe les vues similaires).
    5. Sélection du centroïde de chaque cluster.
    6. Sauvegarde optimisée.
    """
    clip_ctx, model_vis, proc_vis = models_context
    start_time = time.perf_counter()
    
    # Création dossier temporaire/final
    os.makedirs(results_dir, exist_ok=True)

    # 1. Chargement PIL
    pil_images = []
    original_paths = []
    
    for path, b in images_bytes:
        try:
            img = _pil_from_bytes(b)
            pil_images.append(img)
            original_paths.append(path)
        except Exception:
            pass # Image corrompue, on skip

    total_initial = len(pil_images)
    if total_initial == 0:
        return {"listing_id": listing_id, "total": 0, "kept": 0, "selected": 0, "clusters": 0, "paths": ""}

    # 2. Filtrage CLIP
    kept_indices = filter_with_clip(clip_ctx, pil_images, batch_size=16)
    
    # On filtre les listes
    pil_kept = [pil_images[i] for i in kept_indices]
    path_kept = [original_paths[i] for i in kept_indices]
    
    # Nettoyage mémoire des images rejetées
    for i in range(total_initial):
        if i not in kept_indices:
            pil_images[i].close()
            
    if not pil_kept:
        return {"listing_id": listing_id, "total": total_initial, "kept": 0, "selected": 0, "clusters": 0, "paths": ""}

    # 3. Embedding DINO (Le coeur de la similarité)
    embs = embed_images_in_batches(model_vis, proc_vis, pil_kept, batch_size=16)

    # 4. Clustering Agglomératif (Hierarchical)
    # Si on a 1 seule image, pas de clustering possible
    selected_indices_in_kept = []

    if len(pil_kept) == 1:
        selected_indices_in_kept = [0]
        labels = [0]
    else:
        # Distance Threshold : Si distance cosine < threshold, on fusionne.
        # Plus le threshold est petit, plus les images doivent être identiques.
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage='average',
            distance_threshold=similarity_threshold
        )
        labels = clustering.fit_predict(embs)
        unique_labels = np.unique(labels)

        # Pour chaque cluster, on choisit le représentant (Centroïde)
        for label in unique_labels:
            indices = np.where(labels == label)[0]

            if len(indices) == 1:
                selected_indices_in_kept.append(indices[0])
            else:
                # On calcule le vecteur moyen du cluster
                cluster_vectors = embs[indices]
                centroid = np.mean(cluster_vectors, axis=0).reshape(1, -1)

                # On trouve l'image la plus proche mathématiquement de ce "centre"
                dists = cosine_distances(cluster_vectors, centroid).flatten()
                best_idx = indices[np.argmin(dists)]
                selected_indices_in_kept.append(best_idx)

    # 5. Application des contraintes (Min / Max)
    # Note : Le clustering a déjà réduit la redondance. 
    # Si on dépasse max_images, on doit choisir les clusters les plus pertinents.
    # DINO trie souvent implicitement par types. On coupe simplement ici pour l'instant.

    if len(selected_indices_in_kept) > max_images:
        selected_indices_in_kept = selected_indices_in_kept[:max_images]
        
    # 6. Sauvegarde
    saved_paths = []

    for i, idx in enumerate(selected_indices_in_kept):
        pil_img = pil_kept[idx]
        orig_path = path_kept[idx]

        # On renomme proprement : c{cluster_id}_{nom_original}
        cluster_id = labels[idx] if len(pil_kept) > 1 else 0
        filename = f"c{cluster_id}_{os.path.basename(orig_path)}"
        dst_path = os.path.join(results_dir, filename)

        try:
            # Sauvegarde JPEG
            pil_img.save(dst_path, format="JPEG", quality=95, optimize=True)
            saved_paths.append(filename)
        except Exception as e:
            print(f"[{listing_id}] Erreur save {filename}: {e}")

    # Nettoyage final
    for img in pil_kept: img.close()
    
    return {
        "listing_id": listing_id,
        "total": total_initial,
        "kept": len(pil_kept),          # Images saines (pas des plans)
        "selected": len(saved_paths),   # Images finales (clusters uniques)
        "clusters": len(np.unique(labels)) if len(pil_kept) > 1 else 1,
        "paths": "|".join(saved_paths),
        "time_s": time.perf_counter() - start_time
    }
