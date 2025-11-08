import argparse
import os, gc
import platform
import pandas as pd
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.amp import autocast

from transformers import Blip2Processor, Blip2ForConditionalGeneration



# ---------------- utilitaires ----------------
def format_bytes(num_bytes: int) -> str:
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
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Capacité mémoire totale: {format_bytes(torch.cuda.get_device_properties(0).total_memory)}")
        print(f"is_bf16_supported: {torch.cuda.is_bf16_supported()}")
    print("===============================")


def gpu_memory(label: str):
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
    # supprime des colonnes si elles existent (adapte selon ton CSV)
    for col in ['images', 'price']:
        if col in dt.columns:
            dt.drop(columns=[col], inplace=True)
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
    """Trois branches (img, text, tab) -> concat -> final head (2 sorties)."""
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
    parser.add_argument("-csv", "--csv_file", type=str, default="../output/csv/annonces_annonces_Auvergne-Rho╠éne-Alpes_0.csv_0.csv")
    parser.add_argument("-i", "--image_dir", type=str, default="../output/images")
    parser.add_argument("--model_id", type=str, default="Salesforce/blip2-flan-t5-xl")
    args = parser.parse_args()

    system_info()
    final_device = torch.device("cuda")
    print(f"[INFO] Device: {final_device}")

    # --- Préparation des données ---
    data_dict, dt = prepare_input(args.csv_file, args.image_dir)
    text_cols = [c for c in dt.columns if dt[c].dtype == 'object' and c not in ['id']]
    num_cols = [c for c in dt.columns if dt[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    # --- Charger le modèle BLIP2 (torch_dtype si bf16 supporté) ---
    use_bf16 = torch.cuda.is_bf16_supported()
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map={"": "cuda"},
        dtype=torch.bfloat16 if use_bf16 else None,
    )
    model.eval()
    processor = Blip2Processor.from_pretrained(args.model_id, use_fast=True)

    vision = model.vision_model
    text_encoder = model.get_encoder()

    # s'assurer que les sous-modules sont sur GPU
    vision.to(final_device)
    text_encoder.to(final_device)

    # on supprime la référence globale pour libérer mémoire si pas utilisé
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Freeze les poids
    for p in vision.parameters():
        p.requires_grad = False
    for p in text_encoder.parameters():
        p.requires_grad = False

    img_feat_dim = vision.config.hidden_size
    text_feat_dim = getattr(text_encoder.config, "hidden_size", img_feat_dim)
    tab_dim = max(1, len(num_cols))

    # --- Instanciation downstream modules (float32) ---
    fusion_transformer = ImageFusionTransformer(img_feat_dim).to(final_device).eval()
    multimodal_mlp = MultiModalMLP(img_feat_dim, text_feat_dim, tab_dim, hidden=512).to(final_device).eval()

    print(f"[INFO] dims img:{img_feat_dim} text:{text_feat_dim} tab:{tab_dim}")
    gpu_memory("après init")

    # --- Boucle d'inférence ---
    dtype_for_autocast = torch.bfloat16 if use_bf16 else torch.float16
    print(f"[INFO] using autocast dtype: {dtype_for_autocast}")

    for idx, row in dt.iterrows():
        id_str = str(row['id'])
        images = data_dict.get(id_str, [])
        if len(images) == 0:
            continue

        # Encodage inputs (CPU -> on les envoie sur GPU)
        img_inputs = processor(images=images, return_tensors="pt", padding=True)
        img_inputs = {k: v.to(final_device) for k, v in img_inputs.items()}

        text_data = " | ".join([str(row.get(c, "")) for c in text_cols if c in row.index]) if len(text_cols) > 0 else ""
        text_inputs = processor(text=text_data, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(final_device) for k, v in text_inputs.items()}

        # numeric features -> keep float32 (autocast handles mixing)
        num_vals = []
        for c in num_cols:
            try:
                v = row.get(c, 0.0)
                num_vals.append(float(v))
            except Exception:
                num_vals.append(0.0)
        if len(num_vals) == 0:
            num_array = np.zeros((tab_dim,), dtype=np.float32)
        else:
            num_array = np.asarray(num_vals, dtype=np.float32)
        num_feats = torch.from_numpy(num_array).unsqueeze(0).to(device=final_device, dtype=torch.float32)

        # Inference sous autocast (bf16 si supporté)
        with torch.no_grad():
            with autocast('cuda', dtype=dtype_for_autocast):
                # vision forward
                vision_out = vision(pixel_values=img_inputs['pixel_values'])
                per_image_vecs = vision_out.last_hidden_state.mean(dim=1)  # (N_images, dim)

                # fusion expects batch first dimension = 1 (tu fais per_image_vecs.unsqueeze(0) dans ton code)
                tokens_for_fusion = per_image_vecs.unsqueeze(0).to(final_device)  # autocast takes care of dtype

                # fusion
                img_global = fusion_transformer(tokens_for_fusion)  # if fusion_transformer has ops that prefer fp32, autocast will keep them

                # text forward
                text_out = text_encoder(**text_inputs).last_hidden_state
                text_embeds = text_out.mean(dim=1).to(final_device)

                # optionally ensure shapes/dtypes are consistent (autocast handled dtype)
                # forward downstream
                out1, out2 = multimodal_mlp(img_global, text_embeds, num_feats)

        # quick sanity checks
        if torch.isnan(out1).any() or torch.isnan(out2).any():
            print(f"[ERROR] NaN detected for id={id_str} — consider running a float32 debug run")
            # you can break or continue depending on how you want to handle it
            continue

        print(f"[OUTPUT] id={id_str} Prix 1: {float(out1.item()):.2f} | Prix 2: {float(out2.item()):.2f}")


if __name__ == "__main__":
    main()
