import torch
import torch.nn as nn
import torch.nn.functional as F
import timm 
from transformers import AutoModel



# ==============================================================================
# BLOCS DE BASE
# ==============================================================================
class PeriodicEmbedding(nn.Module):
    def __init__(self, embed_dim, sigma=0.1):
        super().__init__()
        # Ici trainable permet au modèle d'ajuster les fréquences optimales.
        self.frequencies = nn.Parameter(torch.randn(embed_dim // 2) * sigma)


    def forward(self, x):
        # x: (Batch, 1) -> freq: (Batch, Dim/2)
        freq = x * self.frequencies.unsqueeze(0)
        # Concat sin/cos -> (Batch, Dim)
        return torch.cat([torch.sin(freq), torch.cos(freq)], dim=-1)


class GEGLU(nn.Module):
    def forward(self, x):
        dim = x.shape[-1] // 2
        return x[..., :dim] * F.gelu(x[..., dim:])


# ==============================================================================
# TOKENIZER TABULAIRE
# ==============================================================================
class SOTAFeatureTokenizer(nn.Module):
    def __init__(self, num_cont, cat_cardinalities, embed_dim):
        super().__init__()

        # 1. Embeddings continus (Neural : Périodique -> Linear)
        self.cont_embeddings = nn.ModuleList([
            nn.Sequential(
                PeriodicEmbedding(embed_dim), 
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(num_cont)
        ])

        # 2. Embeddings catégoriels (Lookup Table)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, embed_dim) for card in cat_cardinalities
        ])

        # 3. Token CLS (Learnable) qui servira à l'agrégation finale
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))


    def forward(self, x_cont, x_cat):
        tokens = []

        # Traitement Continus
        for i, layer in enumerate(self.cont_embeddings):
            # x_cont[:, i] est (Batch,), on veut (Batch, 1) pour le passer au layer
            val = x_cont[:, i].unsqueeze(1)
            # Sortie layer (Batch, Dim) -> Unsqueeze -> (Batch, 1, Dim)
            tokens.append(layer(val).unsqueeze(1))

        # Traitement Catégories
        if x_cat is not None:
            for i, layer in enumerate(self.cat_embeddings):
                # x_cat[:, i] est (Batch,) d'indices long
                tokens.append(layer(x_cat[:, i]).unsqueeze(1))

        # Ajout du CLS Token à la fin
        # Expand pour matcher le batch size : (Batch, 1, Dim)
        cls_tokens = self.cls_token.expand(x_cont.shape[0], -1, -1)
        tokens.append(cls_tokens)

        # Concaténation sur la dimension sequence (dim=1)
        return torch.cat(tokens, dim=1)


# ==============================================================================
# INTERACTION MULTI-MODALE (ATTENTION CROISÉE)
# ==============================================================================
class CrossModalInteraction(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 8), 
            GEGLU(), 
            nn.Dropout(dropout), 
            nn.Linear(dim * 4, dim)
        )


    def forward(self, query_tokens, kv_tokens, key_padding_mask=None, return_attn=False):
        # Query = Tabulaire (C'est lui qu'on met à jour)
        # Key/Value = Contexte (Images + Texte)

        q = self.norm_q(query_tokens)
        kv = self.norm_kv(kv_tokens)

        attn_out, attn_weights = self.multihead_attn(
            query=q, 
            key=kv, 
            value=kv, 
            key_padding_mask=key_padding_mask
        )

        # Connexion résiduelle + FeedForward
        x = query_tokens + attn_out
        x = x + self.ff(self.norm_out(x))

        if return_attn: return x, attn_weights
        return x


# ==============================================================================
# MODÈLE PRINCIPAL
# ==============================================================================
class SOTARealEstateModel(nn.Module):
    def __init__(self, num_continuous, cat_cardinalities, img_model_name='convnext_large.fb_in1k', 
                 text_model_name='almanach/camembert-base', fusion_dim=512, depth=4, freeze_encoders=True):
        super().__init__()

        # --- 1. Vision Encoder ---
        # num_classes=0 supprime la tête de classification finale, renvoie le vecteur de features
        self.img_encoder = timm.create_model(img_model_name, pretrained=True, num_classes=0)
        img_dim = self.img_encoder.num_features
        self.img_proj = nn.Linear(img_dim, fusion_dim)

        # --- 2. Text Encoder ---
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, fusion_dim)

        # --- 3. Tabular Encoder ---
        self.tab_tokenizer = SOTAFeatureTokenizer(num_continuous, cat_cardinalities, fusion_dim)
        # Transformer Encoder pour mixer les features tabulaires entre elles avant la fusion
        self.tab_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=8, dim_feedforward=fusion_dim*4, 
                                       batch_first=True, norm_first=True, activation="gelu"), 
            num_layers=2
        )

        # --- 4. Fusion Layers ---
        self.cross_fusion_layers = nn.ModuleList([
            CrossModalInteraction(fusion_dim) for _ in range(depth)
        ])

        # --- 5. Heads (Prix et Loyer) ---
        self.head_price = nn.Sequential(
            nn.LayerNorm(fusion_dim), 
            nn.Linear(fusion_dim, fusion_dim), 
            GEGLU(), 
            nn.Linear(fusion_dim//2, 1)
        )
        self.head_rent = nn.Sequential(
            nn.LayerNorm(fusion_dim), 
            nn.Linear(fusion_dim, fusion_dim), 
            GEGLU(), 
            nn.Linear(fusion_dim//2, 1)
        )

        # --- 6. Freezing de la Blackbone---
        if freeze_encoders:
            for p in self.img_encoder.parameters(): p.requires_grad = False
            for p in self.text_encoder.parameters(): p.requires_grad = False

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)


    def forward(self, images, image_masks, input_ids, text_mask, x_cont, x_cat=None, return_attn=False):
        """
        images: (B, N_Img, C, H, W)
        image_masks: (B, N_Img) -> 1=Real, 0=Padding
        input_ids, text_mask: (B, Seq_Len)
        x_cont: (B, N_Cont)
        x_cat: (B, N_Cat)
        """
        B, N, C, H, W = images.shape
        device = images.device

        # --- A. Vision ---
        # On aplatit Batch et N_Img pour passer dans l'encodeur 2D standard
        flat_imgs = images.view(B * N, C, H, W)

        # Extraction features (si freeze=True, utiliser no_grad économise de la VRAM)
        with torch.set_grad_enabled(not self.img_encoder.parameters().__next__().requires_grad):
            img_feats = self.img_encoder(flat_imgs) # (B*N, img_dim)

        # Projection et Reshape
        tokens_img = self.img_proj(img_feats).view(B, N, -1) # (B, N, fusion_dim)

        # --- B. Texte ---
        with torch.set_grad_enabled(not self.text_encoder.parameters().__next__().requires_grad):
            txt_out = self.text_encoder(input_ids, attention_mask=text_mask)

        # On prend le CLS token du texte (index 0)
        tokens_txt = self.text_proj(txt_out.last_hidden_state[:, 0, :]).unsqueeze(1) # (B, 1, fusion_dim)

        # --- C. Concaténation du Contexte ---
        context_tokens = torch.cat([tokens_img, tokens_txt], dim=1) # (B, N+1, fusion_dim)

        # --- D. Création du Masque d'Attention Croisée ---
        # PyTorch attention mask: True = IGNORER, False = GARDER
        # image_masks: 1=Garder, 0=Ignorer
        img_padding_mask = (image_masks == 0).to(device) # True là où c'est du padding

        # Le texte (résumé) est toujours présent et pertinent, pas de masque d'exclusion
        txt_padding_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)

        # Masque combiné (B, N+1)
        full_padding_mask = torch.cat([img_padding_mask, txt_padding_mask], dim=1)

        # --- E. Tabulaire (Query) ---
        tokens_tab = self.tab_tokenizer(x_cont, x_cat) # (B, N_Tab + 1, fusion_dim)
        tokens_tab = self.tab_transformer(tokens_tab)

        # --- F. Fusion ---
        attentions = []
        for layer in self.cross_fusion_layers:
            tokens_tab, attn = layer(
                query_tokens=tokens_tab, 
                kv_tokens=context_tokens, 
                key_padding_mask=full_padding_mask, 
                return_attn=True
            )
            attentions.append(attn)

        # On récupère le CLS token tabulaire (le dernier, ajouté par le tokenizer)
        # C'est lui qui a agrégé toute l'info tabulaire + contexte visuel/textuel
        final_vec = tokens_tab[:, -1, :] 

        # --- G. Sortie ---
        # IMPORTANT : On renvoie TOUJOURS le Log-Price.
        log_vente = self.head_price(final_vec)
        log_loc = self.head_rent(final_vec)

        if return_attn:
            return log_vente, log_loc, attentions
        return log_vente, log_loc
