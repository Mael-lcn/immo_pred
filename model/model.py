"""
Ce module définit l'architecture du réseau de neurones hybride optimisé pour le texte bruité.
Mise à jour : Intégration de CamemBERT et stratégie de pooling [CLS].
"""

import torch
import torch.nn as nn
import timm 
from transformers import AutoModel



# --- Module de Projection Tabulaire (Inchangé) ---
class TabularProjector(nn.Module):
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


# --- Modèle Principal  ---
class AdvancedRealEstateModel(nn.Module):
    def __init__(self, 
                 img_model_name='convnext_large.fb_in1k', 
                 text_model_name='almanach/camembert-base',
                 tab_input_dim=10, 
                 fusion_dim=512,
                 freeze_encoders=True):
        """
        Args:
            freeze_encoders (bool): Si True, gèle les experts. 
            Conseil : Dégeler les dernières couches de CamemBERT après quelques époques.
        """
        super().__init__()

        # 1. Expert IMAGE
        print(f"[MODEL] Loading Image Encoder: {img_model_name}")
        self.img_encoder = timm.create_model(img_model_name, pretrained=True, num_classes=0)

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            img_out_dim = self.img_encoder(dummy).shape[-1]

        self.img_proj = nn.Linear(img_out_dim, fusion_dim)

        # 2. Expert TEXTE (CamemBERT)
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
        # Traitement parallèle des N images par annonce
        img_feats = self.img_encoder(images.view(B * N, C, H, W))
        tokens_img = self.img_proj(img_feats).view(B, N, -1)

        # B. Texte (CHANGEMENT CRITIQUE : CLS Token Strategy)
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # On ne fait plus la moyenne. On prend le premier token (Index 0).
        # Dans CamemBERT/RoBERTa, ce token (<s>) sert à classifier toute la phrase.
        # C'est lui qui va apprendre à ignorer le bruit légal à la fin du texte.
        cls_text_state = txt_out.last_hidden_state[:, 0, :] # Shape: [Batch, Hidden_Size]

        token_txt = self.text_proj(cls_text_state).unsqueeze(1) # Shape: [Batch, 1, Fusion_Dim]

        # C. Tabulaire
        token_tab = self.tab_projector(tab_data)

        # D. Fusion
        # On crée une séquence : [CLS_Global, Token_Texte, Token_Tab, Tokens_Images...]
        cls = self.final_cls_token.expand(B, -1, -1)
        sequence = torch.cat([cls, token_txt, token_tab, tokens_img], dim=1)
        
        # Le Transformer mixe toutes les infos
        fused_sequence = self.fusion_transformer(sequence)
        
        # On récupère le vecteur final (celui du CLS global à l'index 0)
        final_vec = fused_sequence[:, 0, :]

        # E. Prédictions (LOG SPACE)
        log_vente = self.head_price(final_vec)
        log_loc = self.head_rent(final_vec)

        # F. Conversion Automatique
        if self.training:
            return log_vente, log_loc # Retourne Log pour la Loss (MSELoss)
        else:
            # Retourne Euros pour l'utilisateur (exp(x) - 1)
            return torch.relu(torch.expm1(log_vente)), torch.relu(torch.expm1(log_loc))
