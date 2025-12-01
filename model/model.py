"""
model.py - Architecture SOTA "Deep Interaction" pour l'Immobilier.

Cette architecture corrige les faiblesses des modèles classiques :
1. Les chiffres (Lat/Lon) sont traités via des 'Periodic Embeddings' (Sin/Cos) pour capter la finesse géographique.
2. La fusion utilise une 'Cross-Attention' : Le Tabulaire "interroge" l'Image et le Texte.
3. Les activations sont des GEGLU (Gated Linear Units), supérieures au ReLU pour le tabulaire.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm 
from transformers import AutoModel



# ==============================================================================
# 1. BRIQUES DE BASE SOTA (Embeddings & Activations)
# ==============================================================================

class PeriodicEmbedding(nn.Module):
    """
    Transforme un scalaire (ex: Latitude) en vecteur haute fréquence.
    Permet au modèle de distinguer 48.85 de 48.86 avec une précision extrême.
    """
    def __init__(self, embed_dim, sigma=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # On crée des fréquences aléatoires fixes (non entraînables au début pour stabilité)
        # Dim // 2 car on a Sin et Cos pour chaque fréquence
        self.frequencies = nn.Parameter(torch.randn(embed_dim // 2) * sigma)


    def forward(self, x):
        # x: [Batch, 1]
        freq = x * self.frequencies.unsqueeze(0) # [Batch, Dim/2]
        # Concaténation Sin/Cos -> [Batch, Dim]
        return torch.cat([torch.sin(freq), torch.cos(freq)], dim=-1)



class GEGLU(nn.Module):
    """
    Gated Linear Unit avec activation GELU.
    SOTA actuel pour les FeedForward Networks tabulaires (cf. papier "GLU Variants").
    """
    def forward(self, x):
        # On coupe le vecteur en deux : une moitié sert de "porte" (gate) pour l'autre
        dim = x.shape[-1] // 2
        return x[..., :dim] * F.gelu(x[..., dim:])


# ==============================================================================
# 2. FEATURE TOKENIZER (Tabulaire)
# ==============================================================================

class SOTAFeatureTokenizer(nn.Module):
    def __init__(self, num_cont, cat_cardinalities, embed_dim):
        super().__init__()
        
        # A. Continus : Periodic Embeddings + Projection linéaire pour mixer
        self.cont_embeddings = nn.ModuleList([
            nn.Sequential(
                PeriodicEmbedding(embed_dim),
                nn.Linear(embed_dim, embed_dim) # Permet d'apprendre des combinaisons de fréquences
            ) for _ in range(num_cont)
        ])
        
        # B. Catégories : Embeddings classiques
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, embed_dim) for card in cat_cardinalities
        ])

        # C. Token CLS "Expert Immobilier"
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))


    def forward(self, x_cont, x_cat):
        tokens = []
        
        # 1. Traitement Périodique des Chiffres
        for i, layer in enumerate(self.cont_embeddings):
            val = x_cont[:, i].unsqueeze(1) # [B, 1]
            tokens.append(layer(val).unsqueeze(1)) # [B, 1, Dim]

        # 2. Traitement des Catégories
        if x_cat is not None and x_cat.shape[1] > 0:
            for i, layer in enumerate(self.cat_embeddings):
                tokens.append(layer(x_cat[:, i]).unsqueeze(1))

        # 3. Ajout du CLS
        B = x_cont.shape[0]
        tokens.append(self.cls_token.expand(B, -1, -1))
        
        return torch.cat(tokens, dim=1) # [B, N_cols+1, Dim]



# ==============================================================================
# 3. MODULE D'INTERACTION (Cross-Attention)
# ==============================================================================

class CrossModalInteraction(nn.Module):
    """
    Permet au flux Tabulaire (Query) d'aller chercher de l'info dans l'Image/Texte (Key/Value).
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

        # FeedForward SOTA avec GEGLU
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 8), # On double la largeur interne pour le split GEGLU
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim) # Retour à la dim originale
        )


    def forward(self, query_tokens, kv_tokens, return_attn=False):
        """
        query_tokens : Le Tabulaire (C'est lui qui cherche l'info)
        kv_tokens    : L'Image + Le Texte (La base de connaissance)
        """
        # 1. Cross-Attention
        q = self.norm_q(query_tokens)
        kv = self.norm_kv(kv_tokens)

        attn_out, attn_weights = self.multihead_attn(query=q, key=kv, value=kv)

        # Connexion Résiduelle 1
        x = query_tokens + attn_out

        # 2. FeedForward (GEGLU)
        x_ff = self.ff(self.norm_out(x))

        # Connexion Résiduelle 2
        x = x + x_ff

        if return_attn:
            return x, attn_weights
        return x



# ==============================================================================
# 4. MODÈLE GLOBAL SOTA
# ==============================================================================
class SOTARealEstateModel(nn.Module):
    def __init__(self, 
                 num_continuous, 
                 cat_cardinalities, 
                 img_model_name='convnext_large.fb_in1k', 
                 text_model_name='almanach/camembert-base',
                 fusion_dim=512,
                 depth=4,
                 freeze_encoders=True):
        
        super().__init__()

        # --- A. ENCODEURS (Experts) ---

        # 1. Vision
        print(f"[SOTA] Vision: {img_model_name}")
        self.img_encoder = timm.create_model(img_model_name, pretrained=True, num_classes=0)
        with torch.no_grad():
            img_dim = self.img_encoder(torch.randn(1, 3, 224, 224)).shape[-1]
        self.img_proj = nn.Linear(img_dim, fusion_dim)

        # 2. Texte
        print(f"[SOTA] Texte: {text_model_name}")
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        txt_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(txt_dim, fusion_dim)

        # 3. Tabulaire (SOTA Tokenizer)
        print(f"[SOTA] Tabulaire: Periodic Embeddings + GEGLU")
        self.tab_tokenizer = SOTAFeatureTokenizer(num_continuous, cat_cardinalities, fusion_dim)

        # Freezing
        if freeze_encoders:
            for p in self.img_encoder.parameters(): p.requires_grad = False
            for p in self.text_encoder.parameters(): p.requires_grad = False

        # --- B. FUSION PROFONDE ---

        # 1. Self-Attention Tabulaire (Pour comprendre les chiffres entre eux)
        # "Si j'ai une grande surface mais que je suis en DPE G..."
        tab_encoder_layer = nn.TransformerEncoderLayer(fusion_dim, nhead=8, dim_feedforward=fusion_dim*4, batch_first=True, norm_first=True)
        self.tab_transformer = nn.TransformerEncoder(tab_encoder_layer, num_layers=2)

        # 2. Cross-Attention Fusion (Le tabulaire va chercher l'info visuelle/textuelle)
        self.cross_fusion_layers = nn.ModuleList([
            CrossModalInteraction(fusion_dim) for _ in range(depth)
        ])

        # --- C. TÊTES DE SORTIE ---
        # On utilise le token CLS tabulaire enrichi pour prédire
        self.head_price = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim * 2), GEGLU(),
            nn.Linear(fusion_dim, 1)
        )
        self.head_rent = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim * 2), GEGLU(),
            nn.Linear(fusion_dim, 1)
        )


    def forward(self, images, input_ids, attention_mask, x_cont, x_cat=None, return_attn=False):
        B, N, C, H, W = images.shape

        # 1. EXTRACTION DES FEATURES (MODALITÉS AUXILIAIRES)
        # Images -> [B, N_imgs, Dim]
        flat_imgs = images.view(B * N, C, H, W)
        img_feats = self.img_encoder(flat_imgs)
        tokens_img = self.img_proj(img_feats).view(B, N, -1)

        # Texte -> [B, 1, Dim]
        txt_out = self.text_encoder(input_ids, attention_mask=attention_mask)
        cls_txt = txt_out.last_hidden_state[:, 0, :]
        token_txt = self.text_proj(cls_txt).unsqueeze(1)

        # On concatène les infos "Auxiliaires" (Context)
        # Context = [Image_1, Image_2, ..., Texte]
        context_tokens = torch.cat([tokens_img, token_txt], dim=1) # [B, N_ctx, Dim]

        # 2. TRAITEMENT TABULAIRE (MODALITÉ PRINCIPALE)
        # Tokens = [Surf, Lat, Lon, DPE, CLS_Tab]
        tokens_tab = self.tab_tokenizer(x_cont, x_cat)

        # Le Tabulaire réfléchit d'abord sur lui-même (Self-Attention)
        tokens_tab = self.tab_transformer(tokens_tab)

        # 3. INTERACTION CROISÉE (FUSION)
        # Le Tabulaire (Query) va lire le Context (Key/Value)
        # À chaque couche, il affine sa compréhension grâce aux images/textes
        attentions = []
        for layer in self.cross_fusion_layers:
            tokens_tab, attn = layer(tokens_tab, context_tokens, return_attn=True)
            attentions.append(attn)
 
        # 4. PRÉDICTION
        # On prend le dernier token du flux tabulaire (le CLS Tabulaire enrichi)
        # C'est lui qui a "vu" tous les chiffres et posé des questions aux images
        final_vec = tokens_tab[:, -1, :] 

        log_vente = self.head_price(final_vec)
        log_loc = self.head_rent(final_vec)

        # 5. SORTIE
        if self.training:
            if return_attn: return log_vente, log_loc, attentions
            return log_vente, log_loc
        else:
            return torch.exp(log_vente), torch.exp(log_loc), attentions if return_attn else None
