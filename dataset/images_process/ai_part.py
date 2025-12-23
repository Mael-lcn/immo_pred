import os
import shutil
import time
from io import BytesIO
from typing import List, Tuple, Dict

from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel, BitsAndBytesConfig
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



# -------------------------
# UTIL / bnb config
# -------------------------
def _make_bnb_config():
    # automatic attempt: 4-bit NF4 float16
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
    compiled = False
    if hasattr(torch, "compile"):
        try:
            print(f"[AI] attempting torch.compile() for {name} ...")
            model = torch.compile(model)
            compiled = True
            print(f"[AI] torch.compile() succeeded for {name}")
        except Exception as e:
            print(f"[AI] torch.compile() failed for {name}: {e}")
    else:
        print("[AI] torch.compile not available in this torch version.")
    return model, compiled


# -------------------------
# Model init
# -------------------------
def init_models(device="cuda",
                clip_hf_id="openai/clip-vit-base-patch32",
                visual_id="facebook/dinov3-vits16-pretrain-lvd1689m"):
    """
    Charge CLIP (HF preferred) et modèle visuel (DINOv3) en tentant la quantization 4-bit NF4.
    Renvoie dicts/objets utiles et prints la config chargée.
    """
    device = device if device in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AI] init_models: device={device}")

    # try prepare bnb config
    bnb_cfg = _make_bnb_config()
    if bnb_cfg is not None:
        print("[AI] BitsAndBytesConfig prepared (4-bit NF4, dtype=float16) — quantization will be attempted.")
    else:
        print("[AI] BitsAndBytesConfig not available in this environment. Quantization will be skipped.")

    # ---- CLIP HF attempt ----
    clip_cfg = {}
    print(f"[AI] Loading HF CLIP {clip_hf_id} (quantize attempted)...")
    proc = CLIPProcessor.from_pretrained(clip_hf_id)
    map_args = {}
    if bnb_cfg is not None:
        map_args["quantization_config"] = bnb_cfg
        map_args["device_map"] = "auto"
    else:
        map_args["device_map"] = device if device != "cpu" else None

    clip_model = CLIPModel.from_pretrained(clip_hf_id, **map_args)
    clip_model.eval()
    clip_model, clip_compiled = try_compile_model(clip_model, name="HF CLIPModel")
    print(f"[AI] HF CLIP loaded: device_map={map_args.get('device_map')}, compiled={clip_compiled}, quantized={(bnb_cfg is not None)}")

    # prepare text prototypes on CPU/device (let HF place shards if sharded)
    device_t = torch.device(device if device != "cpu" else "cpu")
    outdoor_texts = [
        "an outdoor scene", "a garden", "a backyard", "a street", "a city street",
        "a building facade", "an exterior of a house", "an exterior",
        "an empty lot", "a construction site", "a building plot", "a garden area",
        "a yard under construction", "an open land", "a terrain", "a backyard garden",
        "a landscaped garden", "a fenced land", "a vacant land"
    ]
    indoor_texts = [
        "an indoor scene", "an interior", "a living room", "a bedroom", "a kitchen",
        "an apartment interior", "a bathroom", "a hallway", "a dining room"
    ]
    bad_texts = [
        "a logo", "company logo", "logo", "a floor plan", "a blueprint",
        "a map", "a watermark", "a document", "a scanned document",
        "a document photo", "a screenshot", "a business card", "a sign"
    ]
    with torch.inference_mode():
        tokens_out = proc.tokenizer(outdoor_texts, return_tensors="pt", padding=True).to(device_t)
        tokens_in  = proc.tokenizer(indoor_texts, return_tensors="pt", padding=True).to(device_t)
        tokens_bad = proc.tokenizer(bad_texts, return_tensors="pt", padding=True).to(device_t)

        emb_out = clip_model.get_text_features(**tokens_out)
        emb_in  = clip_model.get_text_features(**tokens_in)
        emb_bad = clip_model.get_text_features(**tokens_bad)

        emb_out = F.normalize(emb_out.float(), dim=-1)
        emb_in  = F.normalize(emb_in.float(), dim=-1)
        emb_bad = F.normalize(emb_bad.float(), dim=-1)

    clip_cfg = {
        "backend": "hf",
        "model": clip_model,
        "processor": proc,
        "emb_out": emb_out,
        "emb_in": emb_in,
        "emb_bad": emb_bad,
        "thresholds": {"indoor": 0.20, "bad": 0.30, "outdoor": 0.25},
        "debug": False,
        "device": device,
        "compiled": clip_compiled,
        "quantized": (bnb_cfg is not None)
    }


    # ---- VISUAL (DINOv3) ----
    vis_compiled = False
    vis_quantized = False
    try:
        print(f"[AI] Loading visual model {visual_id} (quant attempt)...")
        bnb_cfg = _make_bnb_config()
        map_args = {}
        if bnb_cfg is not None:
            map_args["quantization_config"] = bnb_cfg
            map_args["device_map"] = "auto"
        else:
            map_args["device_map"] = device if device != "cpu" else None

        processor_vis = AutoImageProcessor.from_pretrained(visual_id)
        model_vis = AutoModel.from_pretrained(visual_id, **map_args)
        model_vis.eval()
        model_vis, vis_compiled = try_compile_model(model_vis, name="VisualModel(DINOv3)")
        vis_quantized = (bnb_cfg is not None)
        print(f"[AI] Visual model loaded. compiled={vis_compiled}, quantized={vis_quantized}, device_map={map_args.get('device_map')}")
    except Exception as e:
        print(f"[AI] Visual AutoModel load with quant args failed: {e}. Trying non-quantized load...")
        processor_vis = AutoImageProcessor.from_pretrained(visual_id)
        model_vis = AutoModel.from_pretrained(visual_id)
        if device != "cpu":
            model_vis = model_vis.to(device)
        model_vis.eval()
        model_vis, vis_compiled = try_compile_model(model_vis, name="VisualModel(DINOv3)-fallback")
        vis_quantized = False
        print(f"[AI] Visual fallback loaded. compiled={vis_compiled}, quantized={vis_quantized}")

    # summary
    print("[AI] Models initialized:")
    print(f"   CLIP backend: {clip_cfg['backend']}, compiled={clip_cfg.get('compiled',False)}, quantized={clip_cfg.get('quantized',False)}")
    print(f"   Visual: id={visual_id}, compiled={vis_compiled}, quantized={vis_quantized}")
    return clip_cfg, model_vis, processor_vis

# -------------------------
# Core: filter (HF) on PIL images and embeddings
# -------------------------
def _pil_from_bytes(b: bytes):
    return Image.open(BytesIO(b)).convert("RGB")


def batch_clip_filter_hf_from_pils(clip_cfg, pil_imgs: List[Tuple[str, Image.Image]], batch_size: int = 16):
    """
    pil_imgs: list of (path, PIL.Image)
    returns list of tuples: (path, keep_bool, sims_dict, pil_image)
    """
    model = clip_cfg["model"]
    proc = clip_cfg["processor"]
    emb_in = clip_cfg["emb_in"]
    emb_out = clip_cfg["emb_out"]
    emb_bad = clip_cfg["emb_bad"]
    t_in = clip_cfg["thresholds"]["indoor"]
    t_bad = clip_cfg["thresholds"]["bad"]
    t_out = clip_cfg["thresholds"]["outdoor"]
    device = clip_cfg["device"]

    results = []
    device_t = torch.device(device if device != "cpu" else "cpu")
    for i in range(0, len(pil_imgs), batch_size):
        batch = pil_imgs[i:i+batch_size]
        paths = [p for p, _ in batch]
        imgs = [im for _, im in batch]
        with torch.inference_mode():
            inputs = proc(images=imgs, return_tensors="pt").to(device_t)
            img_feats = model.get_image_features(**inputs)
            img_feats = F.normalize(img_feats.float(), dim=-1)
            sim_in = (img_feats @ emb_in.T).cpu().numpy()
            sim_out = (img_feats @ emb_out.T).cpu().numpy()
            sim_bad = (img_feats @ emb_bad.T).cpu().numpy()

        for k in range(len(imgs)):
            max_in = float(sim_in[k].max()) if sim_in.shape[1] > 0 else 0.0
            max_out = float(sim_out[k].max()) if sim_out.shape[1] > 0 else 0.0
            max_bad = float(sim_bad[k].max()) if sim_bad.shape[1] > 0 else 0.0
            margin_in, margin_out = 0.08, 0.06
            if (max_bad > max_in + margin_in) or (max_bad > t_bad):
                keep=False; reason="bad"
            elif (max_out > max_in + margin_out) or (max_out > t_out):
                keep=False; reason="outdoor"
            elif (max_in > t_in) or (max_in - max_out > margin_in):
                keep=True; reason="indoor"
            else:
                keep=False; reason="uncertain"
            sims={"indoor":max_in, "outdoor":max_out, "bad":max_bad, "reason":reason}
            results.append((paths[k], keep, sims, imgs[k]))
    return results


def batch_clip_filter_openclip_from_pils(clip_cfg, pil_imgs):
    # fallback single-image open_clip logic (keeps api same)
    model = clip_cfg["model"]
    preprocess = clip_cfg["preprocess"]
    t_in = clip_cfg["thresholds"]["indoor"]
    t_bad = clip_cfg["thresholds"]["bad"]
    t_out = clip_cfg["thresholds"]["outdoor"]
    emb_in = clip_cfg["emb_in"]
    emb_out = clip_cfg["emb_out"]
    emb_bad = clip_cfg["emb_bad"]
    device = clip_cfg["device"]

    results = []
    for path, pil in pil_imgs:
        try:
            img_t = preprocess(pil).unsqueeze(0).to(device)
            with torch.inference_mode():
                img_emb = model.encode_image(img_t).float()
                img_emb = F.normalize(img_emb, dim=-1)
                sim_in = (img_emb @ emb_in.T).cpu().numpy().squeeze()
                sim_out = (img_emb @ emb_out.T).cpu().numpy().squeeze()
                sim_bad = (img_emb @ emb_bad.T).cpu().numpy().squeeze()
            max_in = float(sim_in.max()) if getattr(sim_in, 'size', 1) > 0 else 0.0
            max_out = float(sim_out.max()) if getattr(sim_out, 'size', 1) > 0 else 0.0
            max_bad = float(sim_bad.max()) if getattr(sim_bad, 'size', 1) > 0 else 0.0
            margin_in, margin_out = 0.08, 0.06
            if (max_bad > max_in + margin_in) or (max_bad > t_bad):
                keep=False; reason="bad"
            elif (max_out > max_in + margin_out) or (max_out > t_out):
                keep=False; reason="outdoor"
            elif (max_in > t_in) or (max_in - max_out > margin_in):
                keep=True; reason="indoor"
            else:
                keep=False; reason="uncertain"
            sims={"indoor":max_in,"outdoor":max_out,"bad":max_bad,"reason":reason}
            results.append((path, keep, sims, pil))
        except Exception as e:
            results.append((path, False, {"indoor":0,"outdoor":0,"bad":0,"reason":"open_fail"}, pil))
    return results


def embed_images_in_batches(model, processor, pil_images: List[Image.Image], device='cuda', batch_size=16):
    device_t = torch.device(device if device != "cpu" else "cpu")
    all_embs = []
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(device_t) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs)
        feats = None
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            feats = out.last_hidden_state.mean(dim=1)
        else:
            for v in out.values() if isinstance(out, dict) else []:
                if torch.is_tensor(v):
                    feats = v.mean(dim=1)
                    break
        if feats is None:
            raise RuntimeError("Impossible d'extraire des features du modèle.")
        if feats.dim() > 2:
            feats = feats.flatten(1)
        feats = F.normalize(feats.float(), dim=-1)
        all_embs.append(feats.cpu().numpy())
    if not all_embs:
        return np.zeros((0,0))
    return np.vstack(all_embs)


# -------------------------
# High-level: process listing from bytes (called in main process)
# -------------------------
def process_listing_from_bytes(listing_id: str,
                               images_bytes: List[Tuple[str, bytes]],
                               results_dir: str,
                               clip_cfg,
                               model_vis,
                               processor_vis,
                               device: str = "cuda",
                               batch_size: int = 16,
                               num_rooms: int = 1,
                               max_pca_components: int = 16,
                               clip_debug: bool = False) -> Dict:
    """
    images_bytes: list of (path_on_disk, bytes)
    Returns summary dict and copies selected images into results_dir.
    """
    start = time.perf_counter()
    os.makedirs(results_dir, exist_ok=True)
    # convert bytes to PIL images
    pil_pairs = []
    for path, b in images_bytes:
        try:
            pil = _pil_from_bytes(b)
            pil_pairs.append((path, pil))
        except Exception as e:
            print(f"[{listing_id}] can't open bytes -> PIL {path}: {e}")

    if not pil_pairs:
        return {"listing_id": listing_id, "total": 0, "kept": 0, "selected": 0, "clusters": 0, "selected_paths": []}

    # CLIP filtering (HF or open_clip)
    if clip_cfg["backend"] == "hf":
        filt = batch_clip_filter_hf_from_pils(clip_cfg, pil_pairs, batch_size=batch_size)
    else:
        filt = batch_clip_filter_openclip_from_pils(clip_cfg, pil_pairs)

    kept_entries = [(p, sims, pil) for p, keep, sims, pil in filt if keep]
    total = len(pil_pairs)
    if clip_debug:
        kept_n = len(kept_entries)
        print(f"[{listing_id}] CLIP: total={total}, kept={kept_n}")

    if not kept_entries:
        # cleanup PILs
        for _,_,pil in pil_pairs:
            try: pil.close()
            except: pass
        return {"listing_id": listing_id, "total": total, "kept": 0, "selected": 0, "clusters": 0, "selected_paths": []}

    # prepare lists for embedding
    kept_paths = [p for p,_,_ in kept_entries]
    kept_pils  = [pil for _,_,pil in kept_entries]

    # embeddings (visual model)
    embs = embed_images_in_batches(model_vis, processor_vis, kept_pils, device=device, batch_size=batch_size)
    if embs.size == 0:
        for pil in kept_pils:
            try: pil.close()
            except: pass
        return {"listing_id": listing_id, "total": total, "kept": len(kept_pils), "selected": 0, "clusters": 0, "selected_paths": []}

    # PCA
    n_samples, n_features = embs.shape[0], embs.shape[1]
    n_components_pca = min(max_pca_components, n_features, n_samples)
    n_components_pca = max(1, n_components_pca)
    try:
        pca = PCA(n_components=n_components_pca)
        embs_pca = pca.fit_transform(embs)
    except Exception as e:
        print(f"[{listing_id}] PCA failed: {e}")
        embs_pca = embs

    # clusters: use num_rooms to determine n_clusters (never mix listings)
    try:
        n_clusters = max(1, min(int(num_rooms), len(kept_paths)))
    except Exception:
        n_clusters = max(1, min(1, len(kept_paths)))
    selected = []
    if n_clusters > 0:
        # If large, consider MiniBatchKMeans or faiss; here default KMeans (ok for small N)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(embs_pca)
        for lab in range(n_clusters):
            idxs = np.where(labels == lab)[0]
            if idxs.size == 0:
                continue
            centroid = embs_pca[idxs].mean(axis=0)
            dists = np.linalg.norm(embs_pca[idxs] - centroid, axis=1)
            chosen = idxs[np.argmin(dists)]
            selected.append((kept_paths[chosen], lab))

    # copy selected into results_dir (no 'selected' subfolder)
    selected_paths = []
    for p, lab in selected:
        dst = os.path.join(results_dir, f"cluster_{lab}_{os.path.basename(p)}")
        try:
            # original file path p points to disk — but we only have bytes; to be safe, recreate from kept_pils
            # find pil with that path
            idx = kept_paths.index(p)
            pil = kept_pils[idx]
            pil.save(dst, format="JPEG", quality=95, optimize=True)
            selected_paths.append(dst)
        except Exception as e:
            # fallback: try copying original file if exists
            try:
                if os.path.exists(p):
                    shutil.copyfile(p, dst)
                    selected_paths.append(dst)
            except Exception as e2:
                print(f"[{listing_id}] cannot write selected image {p} -> {e} / {e2}")

    # cleanup PILs
    for pil in kept_pils:
        try: pil.close()
        except: pass

    end = time.perf_counter()
    return {
        "listing_id": listing_id,
        "total": total,
        "kept": len(kept_paths),
        "selected": len(selected_paths),
        "clusters": len(set(l for _, l in selected)),
        "selected_paths": selected_paths,
        "time_s": end - start
    }
