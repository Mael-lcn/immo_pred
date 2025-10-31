# ex python .\filter_images.py -v --min_cluster_size 2 sinon on a rien

import os
import argparse
import json
import shutil
from collections import defaultdict

from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import open_clip
import timm
from torchvision import transforms, models as tv_models

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap
import hdbscan



# Récup les img dans folder
def list_images(folder):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))
    return sorted(paths)



class ClipOutdoorFilter:
    def __init__(self, device='cpu', model_name='ViT-B-32', pretrained='openai', threshold=0.25, debug=False):
        print(f"[ClipOutdoorFilter] init: device={device}, model={model_name}, pretrained={pretrained}, threshold={threshold}")
        self.device = device
        # handle different return signatures of create_model_and_transforms
        ret = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        if isinstance(ret, tuple) and len(ret) == 2:
            self.model, self.preprocess = ret
        elif isinstance(ret, tuple) and len(ret) == 3:
            # sometimes returns (model, _, preprocess)
            self.model, _, self.preprocess = ret
        else:
            # fallback to separate factory functions
            try:
                self.model = open_clip.create_model(model_name, pretrained=pretrained)
                self.preprocess = open_clip.transforms.ImageTransform(model_name)
            except Exception:
                # best-effort: try to assign ret directly (may raise later)
                self.model, self.preprocess = ret

        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(device).eval()
        print("[ClipOutdoorFilter] open_clip model loaded successfully")

        # prompts
        self.outdoor_texts = [
            "an outdoor scene", "a garden", "a backyard", "a street",
            "a city street", "a building facade", "an exterior of a house", "an exterior"
        ]
        self.indoor_texts = [
            "an indoor scene", "an interior", "a living room", "a bedroom", "a kitchen", "an apartment interior"
        ]
        self.threshold = threshold
        self.debug = debug

        # Pre-encode text embeddings
        with torch.no_grad():
            tokens_out = self.tokenizer(self.outdoor_texts).to(device)
            tokens_in = self.tokenizer(self.indoor_texts).to(device)
            emb_out = self.model.encode_text(tokens_out).float()
            emb_in = self.model.encode_text(tokens_in).float()
            emb_out /= emb_out.norm(dim=-1, keepdim=True)
            emb_in /= emb_in.norm(dim=-1, keepdim=True)
            self.text_outdoor_emb = emb_out
            self.text_indoor_emb = emb_in
            print(f"[ClipOutdoorFilter] encoded {len(self.outdoor_texts)} outdoor prompts and {len(self.indoor_texts)} indoor prompts")

    def is_outdoor(self, pil_image, print_debug=False):
        x = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_emb = self.model.encode_image(x).float()
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
            sim_out = (img_emb @ self.text_outdoor_emb.T).max().item()
            sim_in = (img_emb @ self.text_indoor_emb.T).max().item()
            decision = False
            reason = ''
            # primary rule
            if sim_out - sim_in > 0.05:
                decision = True
                reason = 'outdoor_sim_minus_indoor_sim > 0.05'
            elif sim_out > self.threshold:
                decision = True
                reason = f'sim_out > threshold ({sim_out:.3f} > {self.threshold})'
            else:
                decision = False
                reason = f'sim_out <= threshold ({sim_out:.3f} <= {self.threshold}) and diff <= 0.05'
            if print_debug or self.debug:
                print(f"[ClipOutdoorFilter] sims -> out: {sim_out:.4f}, in: {sim_in:.4f} -> decision: {'OUTDOOR' if decision else 'INDOOR'} ({reason})")
            return decision, sim_out, sim_in



# Embedding extractor (DINO or fallback)
class VisualEmbedder:
    def __init__(self, device='cpu', model_name='dino_vitbase16', img_size=224, use_bfloat16=False, use_compile=False):
        print(f"[VisualEmbedder] init: device={device}, model={model_name}, img_size={img_size}, use_bfloat16={use_bfloat16}, use_compile={use_compile}")
        self.device = device
        self.img_size = img_size
        self.use_bfloat16 = use_bfloat16
        self.use_compile = use_compile

        model = None
        for candidate in [model_name, 'dino_vitbase16', 'vit_base_patch16_224', 'tf_efficientnet_b3_ns']:
            try:
                print(f"[VisualEmbedder] trying to load model: {candidate}")
                model = timm.create_model(candidate, pretrained=True)
                print(f"[VisualEmbedder] loaded model: {candidate}")
                break
            except Exception as e:
                print(f"[VisualEmbedder] fail loading {candidate}: {e}")

        if model is None:
            print('[VisualEmbedder] timm failed to load candidates — falling back to torchvision.resnet50')
            model = tv_models.resnet50(pretrained=True)
            model.fc = torch.nn.Identity()
            print('[VisualEmbedder] loaded torchvision.resnet50 as fallback')

        self.model = model.eval().to(device)

        try:
            if hasattr(self.model, 'reset_classifier'):
                self.model.reset_classifier(0)
                print('[VisualEmbedder] reset classifier head (if present)')
        except Exception as e:
            print('[VisualEmbedder] reset_classifier failed:', e)

        # compile if requested and supported
        if use_compile:
            try:
                print('[VisualEmbedder] attempting torch.compile()')
                self.model = torch.compile(self.model)
                print('[VisualEmbedder] torch.compile succeeded')
            except Exception as e:
                print('[VisualEmbedder] torch.compile FAILED:', e)
                print('[VisualEmbedder] continuing without compile')

        # preprocess transform
        self.preprocess = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def embed(self, pil_image, print_debug=False):
        x = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        use_amp = self.use_bfloat16 and torch.cuda.is_available()
        if use_amp:
            ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            ctx = torch.no_grad()
        with ctx:
            with torch.no_grad():
                try:
                    feat = self.model(x)
                except Exception as e:
                    print('[VisualEmbedder] model forward FAILED:', e)
                    try:
                        feat = self.model(x.float())
                        print('[VisualEmbedder] retry forward with float successful')
                    except Exception as e2:
                        print('[VisualEmbedder] retry failed:', e2)
                        raise

        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        feat = feat.reshape(feat.shape[0], -1)
        arr = feat.cpu().numpy().squeeze()
        norm = np.linalg.norm(arr) + 1e-9
        arr = arr / norm
        if print_debug:
            print(f"[VisualEmbedder] produced embedding of shape {arr.shape}, norm(before)={norm:.4f}")
        return arr

# --------------------
# Pipeline orchestration
# --------------------
def run_pipeline(args):
    print('\n=== PIPELINE START ===')
    print(f"Input dir: {args.input_dir}\nOutput dir: {args.output_dir}\nDevice: {args.device}\nUse BF16: {args.use_bfloat16}\nUse torch.compile: {args.use_torch_compile}\nVerbose: {args.verbose}")
    os.makedirs(args.output_dir, exist_ok=True)
    image_paths = list_images(args.input_dir)
    print(f"Found {len(image_paths)} images in {args.input_dir}")

    # CLIP filter
    clip_filter = None
    try:
        clip_filter = ClipOutdoorFilter(device=args.device, model_name=args.clip_model, pretrained=args.clip_pretrained, threshold=args.clip_threshold, debug=args.verbose)
    except Exception as e:
        print('[run_pipeline] ClipOutdoorFilter init failed (continuing without CLIP):', e)

    kept = []
    dropped = []
    print_every = 50
    for i, p in enumerate(tqdm(image_paths, desc='Filtering outdoor')):
        try:
            img = Image.open(p).convert('RGB')
        except Exception as e:
            print('[run_pipeline] Could not open', p, e)
            continue
        if clip_filter is not None:
            try:
                is_out, sim_out, sim_in = clip_filter.is_outdoor(img, print_debug=(args.verbose and (i % print_every == 0)))
            except Exception as e:
                print('[run_pipeline] clip_filter failed for', p, e)
                is_out = False
                sim_out = sim_in = 0.0
            if is_out:
                dropped.append((p, sim_out, sim_in))
                if args.verbose and (i % print_every == 0):
                    print(f"[run_pipeline] DROPPED (outdoor) {p} -> sim_out={sim_out:.3f}, sim_in={sim_in:.3f}")
            else:
                kept.append(p)
                if args.verbose and (i % print_every == 0):
                    print(f"[run_pipeline] KEPT {p} -> sim_out={sim_out:.3f}, sim_in={sim_in:.3f}")
        else:
            kept.append(p)
    print(f"Filtering done. kept={len(kept)}, dropped={len(dropped)}")

    if len(kept) == 0:
        print('[run_pipeline] No images kept after filtering - exiting')
        return

    # Embeddings
    try:
        embedder = VisualEmbedder(device=args.device, model_name=args.embed_model, img_size=args.img_size, use_bfloat16=args.use_bfloat16, use_compile=args.use_torch_compile)
    except Exception as e:
        print('[run_pipeline] Failed to init VisualEmbedder:', e)
        return

    embs = []
    kept_imgs = []
    for i, p in enumerate(tqdm(kept, desc='Embedding images')):
        try:
            img = Image.open(p).convert('RGB')
            e = embedder.embed(img, print_debug=(args.verbose and (i % 50 == 0)))
            embs.append(e)
            kept_imgs.append(p)
            if args.verbose and (i % 200 == 0):
                print(f"[run_pipeline] Embedded {i+1}/{len(kept)}: {p}")
        except Exception as e:
            print('[run_pipeline] embed fail for', p, str(e))

    embs = np.vstack(embs)
    print('Embeddings complete. shape=', embs.shape)

    # PCA
    n_samples, n_features = embs.shape
    n_pca = min(256, n_samples, n_features)
    if n_pca < 1:
        print('[run_pipeline] Not enough dimensions/samples for PCA. Exiting.')
        return
    print(f'[run_pipeline] PCA -> n_components={n_pca} (n_samples={n_samples}, n_features={n_features})')
    pca = PCA(n_components=n_pca)
    embs_pca = pca.fit_transform(embs)
    print(f'[run_pipeline] PCA done. explained_variance_ratio_sum={np.sum(pca.explained_variance_ratio_):.3f}')

    # UMAP (safe parameters for small datasets)
    if n_samples < 3:
        print('[run_pipeline] Too few samples for UMAP (n_samples<3). Using PCA output as reduced features.')
        embs_reduced = embs_pca
    else:
        umap_n = min(64, n_samples - 1, embs_pca.shape[1])
        n_neighbors = min(15, max(2, n_samples - 1))
        print(f'[run_pipeline] UMAP params -> n_components={umap_n}, n_neighbors={n_neighbors}')
        try:
            reducer = umap.UMAP(n_components=umap_n, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
            embs_reduced = reducer.fit_transform(embs_pca)
            print('UMAP done. reduced shape=', embs_reduced.shape)
        except Exception as e:
            print('[run_pipeline] UMAP failed, falling back to PCA output:', e)
            embs_reduced = embs_pca

    # Clustering
    print('[run_pipeline] Starting clustering...')
    labels = None
    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size)
        labels = clusterer.fit_predict(embs_reduced)
        n_clusters = len(set(labels)) - (-1 in labels)
        print(f'[run_pipeline] HDBSCAN produced {n_clusters} clusters')
    except Exception as e:
        print('[run_pipeline] HDBSCAN failed, fallback to KMeans:', e)
        k = max(2, len(embs_reduced) // 5)
        clusterer = KMeans(n_clusters=k, random_state=42)
        labels = clusterer.fit_predict(embs_reduced)
        print(f'[run_pipeline] KMeans fallback produced {k} clusters')

    clusters = defaultdict(list)
    for idx, lab in enumerate(labels):
        clusters[lab].append(idx)
    print(f'[run_pipeline] number of clusters (including noise): {len(clusters)}')

    # Selection: nearest to centroid
    selected = []
    export_rows = []
    for lab, idxs in clusters.items():
        if lab == -1:
            if args.verbose:
                print(f"[run_pipeline] skipping noise cluster with {len(idxs)} items")
            continue
        pts = embs_reduced[idxs]
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid[None, :], axis=1)
        order = np.argsort(dists)
        take = min(args.n_per_cluster, len(idxs))
        for i in order[:take]:
            sel_idx = idxs[i]
            selected.append((kept_imgs[sel_idx], lab))
            if args.export_csv:
                export_rows.append((kept_imgs[sel_idx], lab, float(dists[i])))
        if args.verbose:
            print(f"[run_pipeline] cluster {lab}: size={len(idxs)}, selected={take}")

    print('[run_pipeline] total selected images=', len(selected))

    # Save selected images
    sel_dir = os.path.join(args.output_dir, 'selected')
    os.makedirs(sel_dir, exist_ok=True)
    cluster_counts = {}
    for p, lab in selected:
        cluster_dir = os.path.join(sel_dir, f'cluster_{lab}')
        os.makedirs(cluster_dir, exist_ok=True)
        fname = os.path.basename(p)
        dst = os.path.join(cluster_dir, fname)
        try:
            shutil.copyfile(p, dst)
            cluster_counts[lab] = cluster_counts.get(lab, 0) + 1
        except Exception as e:
            print('[run_pipeline] failed to copy', p, e)

    # Save metadata
    meta = {
        'total_images': len(image_paths),
        'kept': len(kept),
        'dropped': len(dropped),
        'selected_count': len(selected),
        'cluster_counts': cluster_counts
    }
    with open(os.path.join(args.output_dir, 'summary_verbose.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print('Summary saved to', os.path.join(args.output_dir, 'summary_verbose.json'))

    # Export CSV if requested
    if args.export_csv and len(export_rows) > 0:
        import csv
        csv_path = os.path.join(args.output_dir, 'selected_embeddings.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.writer(cf)
            writer.writerow(['image_path', 'cluster', 'dist_to_centroid'])
            for r in export_rows:
                writer.writerow(r)
        print('CSV exported to', csv_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',  default='../output/test_img/2507940269')
    parser.add_argument('--output_dir', default='../output/cluster_res/')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--clip_model', default='ViT-B-32')
    parser.add_argument('--clip_pretrained', default='openai')
    parser.add_argument('--clip_threshold', type=float, default=0.25)
    parser.add_argument('--embed_model', default='dino_vitbase16')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--n_per_cluster', type=int, default=3)
    parser.add_argument('--min_cluster_size', type=int, default=30)
    parser.add_argument('--use_bfloat16', action='store_true', dest='use_bfloat16', help='Use bfloat16 autocast during embedding if available')
    parser.add_argument('--use_torch_compile', action='store_true', dest='use_torch_compile', help='Try torch.compile() on the embedder model (may fail on some ops)')
    parser.add_argument('--export_csv', action='store_true', dest='export_csv', help='Export CSV with selected images and distances')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    print('\n=== ENV INFO ===')
    print('torch version:', torch.__version__)
    print('cuda available:', torch.cuda.is_available())
    try:
        if torch.cuda.is_available():
            print('cuda device name:', torch.cuda.get_device_name(0))
            print('cuda device count:', torch.cuda.device_count())
    except Exception as e:
        print('cuda device info error:', e)
    print('=== END ENV INFO ===\n')

    run_pipeline(args)


if __name__ == '__main__':
    main()
