import torch
import torch.nn as nn
import torch.nn.functional as F
import timm 
from transformers import AutoModel



class PeriodicEmbedding(nn.Module):
    def __init__(self, embed_dim, sigma=0.1):
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(embed_dim // 2) * sigma)
    def forward(self, x):
        freq = x * self.frequencies.unsqueeze(0)
        return torch.cat([torch.sin(freq), torch.cos(freq)], dim=-1)


class GEGLU(nn.Module):
    def forward(self, x):
        dim = x.shape[-1] // 2
        return x[..., :dim] * F.gelu(x[..., dim:])


# ==============================================================================
# TOKENIZER (Mise à jour pour binaires en tant que cat)
# ==============================================================================
class SOTAFeatureTokenizer(nn.Module):
    def __init__(self, num_cont, cat_cardinalities, embed_dim):
        super().__init__()
        # Embeddings périodiques UNIQUEMENT pour les variables continues (Surface, Lat, Lon)
        self.cont_embeddings = nn.ModuleList([
            nn.Sequential(PeriodicEmbedding(embed_dim), nn.Linear(embed_dim, embed_dim)) 
            for _ in range(num_cont)
        ])
        # Embeddings classiques pour Catégories ET Binaires
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, embed_dim) for card in cat_cardinalities
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))


    def forward(self, x_cont, x_cat):
        tokens = []
        # Continus
        for i, layer in enumerate(self.cont_embeddings):
            tokens.append(layer(x_cont[:, i].unsqueeze(1)).unsqueeze(1))
        # Catégories
        if x_cat is not None:
            for i, layer in enumerate(self.cat_embeddings):
                tokens.append(layer(x_cat[:, i]).unsqueeze(1))
        
        # CLS
        tokens.append(self.cls_token.expand(x_cont.shape[0], -1, -1))
        return torch.cat(tokens, dim=1)


# ==============================================================================
# INTERACTION AVEC MASKING
# ==============================================================================
class CrossModalInteraction(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 8), GEGLU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )


    def forward(self, query_tokens, kv_tokens, key_padding_mask=None, return_attn=False):
        # query_tokens = Tabulaire
        # kv_tokens = Images + Texte
        # key_padding_mask = BoolTensor où True indique les valeurs à IGNORER (Padding)

        q = self.norm_q(query_tokens)
        kv = self.norm_kv(kv_tokens)

        attn_out, attn_weights = self.multihead_attn(
            query=q, key=kv, value=kv, 
            key_padding_mask=key_padding_mask
        )

        x = query_tokens + attn_out
        x = x + self.ff(self.norm_out(x))

        if return_attn: return x, attn_weights
        return x


class SOTARealEstateModel(nn.Module):
    def __init__(self, num_continuous, cat_cardinalities, img_model_name='convnext_large.fb_in1k', 
                 text_model_name='almanach/camembert-base', fusion_dim=512, depth=4, freeze_encoders=True):
        super().__init__()

        # 1. Vision
        self.img_encoder = timm.create_model(img_model_name, pretrained=True, num_classes=0)
        img_dim = self.img_encoder.num_features
        self.img_proj = nn.Linear(img_dim, fusion_dim)

        # 2. Texte
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, fusion_dim)

        # 3. Tabulaire
        self.tab_tokenizer = SOTAFeatureTokenizer(num_continuous, cat_cardinalities, fusion_dim)
        self.tab_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(fusion_dim, nhead=8, dim_feedforward=fusion_dim*4, batch_first=True, norm_first=True), 
            num_layers=2
        )

        # Fusion
        self.cross_fusion_layers = nn.ModuleList([CrossModalInteraction(fusion_dim) for _ in range(depth)])

        # Têtes
        self.head_price = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, fusion_dim), GEGLU(), nn.Linear(fusion_dim//2, 1))
        self.head_rent = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, fusion_dim), GEGLU(), nn.Linear(fusion_dim//2, 1))

        if freeze_encoders:
            for p in self.img_encoder.parameters(): p.requires_grad = False
            for p in self.text_encoder.parameters(): p.requires_grad = False


    def forward(self, images, image_masks, input_ids, text_mask, x_cont, x_cat=None, return_attn=False):
            B, N, C, H, W = images.shape

            # --- A. Vision ---
            flat_imgs = images.view(B * N, C, H, W)
            img_feats = self.img_encoder(flat_imgs) 
            tokens_img = self.img_proj(img_feats).view(B, N, -1) 

            # --- B. Texte ---
            txt_out = self.text_encoder(input_ids, attention_mask=text_mask)
            tokens_txt = self.text_proj(txt_out.last_hidden_state[:, 0, :]).unsqueeze(1)

            # --- C. Context & Masques ---
            context_tokens = torch.cat([tokens_img, tokens_txt], dim=1)

            # Masquage du padding (True = Ignorer)
            img_padding_mask = (image_masks == 0) 
            txt_padding_mask = torch.zeros((B, 1), dtype=torch.bool, device=images.device)
            full_padding_mask = torch.cat([img_padding_mask, txt_padding_mask], dim=1)

            # --- D. Tabulaire ---
            tokens_tab = self.tab_tokenizer(x_cont, x_cat)
            tokens_tab = self.tab_transformer(tokens_tab)

            # --- E. Fusion ---
            attentions = []
            for layer in self.cross_fusion_layers:
                tokens_tab, attn = layer(tokens_tab, context_tokens, key_padding_mask=full_padding_mask, return_attn=True)
                attentions.append(attn)

            final_vec = tokens_tab[:, -1, :] 
            
            # --- F. Sortie Intelligente ---
            log_vente = self.head_price(final_vec)
            log_loc = self.head_rent(final_vec)

            # LOGIQUE DE BASCULE AUTOMATIQUE
            if self.training:
                # EN TRAIN : On renvoie les LOGS pour que la Loss fonctionne 
                # (car les targets dans le dataloader sont des logs)
                return log_vente, log_loc, attentions if return_attn else None
            else:
                # EN EVAL / PROD : On renvoie les EUROS pour l'utilisateur
                # (On applique l'exponentielle ici)
                return torch.exp(log_vente), torch.exp(log_loc), attentions if return_attn else None
