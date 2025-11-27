"""
Ce module définit l'architecture du réseau de neurones hybride.
Spécificité : Il s'entraîne dans l'espace Logarithmique (stable) mais
prédit en Euros (lisible) lorsqu'il est en mode évaluation.
"""

import torch
import torch.nn as nn
import timm 
from transformers import AutoModel



# --- Module de Projection Tabulaire ---
class TabularProjector(nn.Module):
    """
    Transforme les colonnes numériques (surface, pièces...) en embeddings riches.
    """
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))


    def forward(self, x):
        feat = self.proj(x).unsqueeze(1)
        return feat + self.cls_token 



# --- Modèle Principal ---
class AdvancedRealEstateModel(nn.Module):
    def __init__(self, 
                 img_model_name='convnext_large.fb_in1k', 
                 text_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                 tab_input_dim=10, 
                 fusion_dim=512,
                 freeze_encoders=True):
        """
        Args:
            freeze_encoders (bool): Si True, gèle ConvNeXt et BERT (Indispensable au début).
        """
        super().__init__()

        # 1. Expert IMAGE
        print(f"[MODEL] Loading Image Encoder: {img_model_name}")
        self.img_encoder = timm.create_model(img_model_name, pretrained=True, num_classes=0)

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            img_out_dim = self.img_encoder(dummy).shape[-1]

        self.img_proj = nn.Linear(img_out_dim, fusion_dim)

        # 2. Expert TEXTE
        print(f"[MODEL] Loading Text Encoder: {text_model_name}")
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_out_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_out_dim, fusion_dim)

        # GESTION DU FREEZING
        if freeze_encoders:
            print("[MODEL] Experts are FROZEN ❄️")
            for p in self.img_encoder.parameters(): p.requires_grad = False
            for p in self.text_encoder.parameters(): p.requires_grad = False
        else:
            print("[MODEL] Experts are UNLOCKED 🔥")

        # 3. Expert TABULAIRE
        self.tab_projector = TabularProjector(tab_input_dim, fusion_dim)

        # 4. FUSION (Transformer)
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=8, dim_feedforward=fusion_dim*4, 
            batch_first=True, norm_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(fusion_layer, num_layers=3)
        self.final_cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim))

        # 5. HEADS (Sorties)
        self.head_price = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, 256), nn.GELU(), nn.Linear(256, 1)
        )
        self.head_rent = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, 256), nn.GELU(), nn.Linear(256, 1)
        )


    def forward(self, images, input_ids, attention_mask, tab_data):
        B, N, C, H, W = images.shape

        # A. Images (Batch * N_imgs)
        img_feats = self.img_encoder(images.view(B * N, C, H, W))
        tokens_img = self.img_proj(img_feats).view(B, N, -1)

        # B. Texte (Mean Pooling)
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = txt_out.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        txt_vec = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        token_txt = self.text_proj(txt_vec).unsqueeze(1)

        # C. Tabulaire
        token_tab = self.tab_projector(tab_data)

        # D. Fusion
        cls = self.final_cls_token.expand(B, -1, -1)
        sequence = torch.cat([cls, token_txt, token_tab, tokens_img], dim=1)
        fused_sequence = self.fusion_transformer(sequence)
        final_vec = fused_sequence[:, 0, :]

        # E. Prédictions (LOG SPACE)
        log_vente = self.head_price(final_vec)
        log_loc = self.head_rent(final_vec)

        # F. Conversion Automatique
        if self.training:
            return log_vente, log_loc # Retourne Log pour la Loss
        else:
            # Retourne Euros pour l'utilisateur (exp(x) - 1)
            return torch.relu(torch.expm1(log_vente)), torch.relu(torch.expm1(log_loc))
