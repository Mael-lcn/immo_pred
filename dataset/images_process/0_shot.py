import argparse
import os
import platform
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.amp import autocast

from transformers import Blip2ForConditionalGeneration, Blip2Processor



# ---------------- utilitaires ----------------
def format_bytes(num_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def system_info():
    print("===== Informations système =====")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Capacité mémoire totale: {format_bytes(torch.cuda.get_device_properties(0).total_memory)}")
            print(f"is_bf16_supported: {torch.cuda.is_bf16_supported()}")
        except Exception:
            pass
    print("===============================")


def gpu_memory(label):
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated(0)
        mem_reserved = torch.cuda.memory_reserved(0)
        print(f"[GPU] {label}: {format_bytes(mem_used)} utilisée / {format_bytes(mem_reserved)} réservée")


def load_images(img_dir):
    if not os.path.isdir(img_dir):
        return []

    valid_ext = (".jpg", ".jpeg", ".png")
    imgs = [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for fname in os.listdir(img_dir)
        if fname.lower().endswith(valid_ext)
        for img in [cv2.imread(os.path.join(img_dir, fname), cv2.IMREAD_COLOR)]
        if img is not None
    ]
    return imgs


def prepare_input(csv_path, img_base_dir):
    dt = pd.read_csv(csv_path, sep=";", quotechar='"')
    dt.drop(columns=['images', 'price'], inplace=True)
    dt = dt[:10]
    return {str(r['id']): load_images(os.path.join(img_base_dir, str(r['id']))) for _, r in dt.iterrows()}, dt



# ---------------- modules ----------------
class ImageFusionTransformer(nn.Module):
    def __init__(self, dim, num_layers=2, num_heads=4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim*2, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, tokens):
        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        out = self.enc(x)
        return out[:, 0, :]


class MultiModalMLP(nn.Module):
    def __init__(self, img_dim, text_dim, tab_dim, hidden=512):
        super().__init__()
        self.img_fc = nn.Sequential(nn.Linear(img_dim, hidden), nn.ReLU())
        self.text_fc = nn.Sequential(nn.Linear(text_dim, hidden), nn.ReLU())
        self.tab_fc = nn.Sequential(nn.Linear(tab_dim, hidden), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU())
        self.head1 = nn.Linear(hidden, 1)
        self.head2 = nn.Linear(hidden, 1)

    def forward(self, img_vec, text_vec, tab_vec):
        h_img = self.img_fc(img_vec)
        h_text = self.text_fc(text_vec)
        h_tab = self.tab_fc(tab_vec)
        h = torch.cat([h_img, h_text, h_tab], dim=1)
        h = self.fusion(h)
        return self.head1(h), self.head2(h)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv", "--csv_file", type=str, default="../output/csv/annonces_annonces_Auvergne-Rhone-Alpes_0_csv_0.csv")
    parser.add_argument("-i", "--image_dir", type=str, default="../output/images")
    parser.add_argument("--model_id", type=str, default="Salesforce/blip2-flan-t5-xl")
    args = parser.parse_args()

    system_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    data_dict, dt = prepare_input(args.csv_file, args.image_dir)
    text_cols = [c for c in dt.columns if dt[c].dtype == 'object' and c not in ['id']]
    num_cols = [c for c in dt.columns if dt[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    tab_dim = max(1, len(num_cols))

    use_bf16 = torch.cuda.is_bf16_supported()
    dtype_load = torch.bfloat16 if use_bf16 else torch.float32
    dtype_for_autocast = torch.bfloat16 if use_bf16 else torch.float32

    print("[INFO] Chargement BLIP-2 (device_map='auto')...")
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_id, device_map="auto",
        dtype=dtype_load)

    model.eval()
    processor = Blip2Processor.from_pretrained(args.model_id, use_fast=True)

    # freeze vision/qformer
    for p in model.vision_model.parameters():
        p.requires_grad = False
    for p in model.qformer.parameters():
        p.requires_grad = False
    if hasattr(model, "qformer_projector"):
        for p in model.qformer_projector.parameters():
            p.requires_grad = False
    if hasattr(model, "text_projection"):
        try:
            for p in model.text_projection.parameters():
                p.requires_grad = False
        except Exception:
            pass

    # --- Dimensions embeddings ---
    dummy_pixel = torch.zeros(1, 3, 224, 224).to(device)
    img_feat_model_dim = model.get_image_features(dummy_pixel).shape[-1]
    img_feat_dim = 768  # dimension fusion/MLP
    text_feat_dim = model.text_projection.out_features if hasattr(model, "text_projection") else img_feat_dim

    # projection pour harmonisation
    img_proj = nn.Linear(img_feat_model_dim, img_feat_dim).to(device).eval()

    # downstream modules sur GPU
    fusion_transformer = ImageFusionTransformer(img_feat_dim).to(device).eval()
    multimodal_mlp = MultiModalMLP(img_feat_dim, text_feat_dim, tab_dim, hidden=512).to(device).eval()

    print(f"[INFO] dims img:{img_feat_dim} text:{text_feat_dim} tab:{tab_dim}")
    gpu_memory("après init")
    print(f"[INFO] using autocast dtype: {dtype_for_autocast}")

    # --- Boucle inference ---
    for _, row in dt.iterrows():
        id_str = str(row['id'])
        images = data_dict.get(id_str, [])
        if len(images) == 0:
            continue

        # --- Prépare images ---
        img_inputs = processor(images=images, return_tensors="pt", padding=True)
        pixel_values = img_inputs['pixel_values'].to(device)

        # Text
        text_data = " | ".join([str(row.get(c, "")) for c in text_cols if c in row.index]) if len(text_cols) > 0 else ""
        text_inputs = processor(text=text_data, return_tensors="pt", padding=True, truncation=True)

        # --- Features numériques ---
        num_vals = [float(row.get(c, 0.0)) for c in num_cols] if num_cols else [0.0]*tab_dim
        num_feats = torch.tensor(num_vals, dtype=torch.float32, device=device).unsqueeze(0)

        # --- Extraction features ---
        with torch.no_grad():
            # GPU: images
            with autocast(device_type='cuda', dtype=dtype_for_autocast):
                image_feats = model.get_image_features(pixel_values=pixel_values)
                per_image_vecs = image_feats.mean(dim=1)
                tokens_for_fusion = img_proj(per_image_vecs).unsqueeze(0)
                img_global = fusion_transformer(tokens_for_fusion).to(dtype=torch.float32)

            # Texte
            try:
                text_feats = model.get_text_features(**text_inputs)
                text_embeds = text_feats.to(device=device, dtype=torch.float32)
            except Exception:
                text_embeds = torch.zeros((1, text_feat_dim), device=device, dtype=torch.float32)

            # Forward downstream
            out1, out2 = multimodal_mlp(img_global, text_embeds, num_feats)

        if torch.isnan(out1).any() or torch.isnan(out2).any():
            print(f"[ERROR] NaN detected for id={id_str}")
            continue

        print(f"[OUTPUT] id={id_str} Prix vente: {float(out1.item()):.2f} | Prix location: {float(out2.item()):.2f}")


if __name__ == '__main__':
    main()
