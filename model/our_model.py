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
    """
    Convertit un nombre d'octets en une chaîne lisible (B, KB, MB, GB, TB).

    Args:
        num_bytes (int/float): La taille en octets.

    Returns:
        str: La taille formatée avec l'unité appropriée.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def system_info():
    """
    Affiche les informations techniques sur l'environnement d'exécution
    (Version Python, PyTorch, disponibilité CUDA et détails GPU).
    """
    print("===== Informations système =====")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Capacité mémoire totale: {format_bytes(torch.cuda.get_device_properties(0).total_memory)}")
            # Vérifie si le GPU supporte le format bfloat16 (souvent utilisé pour les LLMs récents)
            print(f"is_bf16_supported: {torch.cuda.is_bf16_supported()}")
        except Exception:
            pass
    print("===============================")


def gpu_memory(label):
    """
    Affiche la mémoire GPU allouée et réservée à un instant T.
    Utile pour le débogage des erreurs OOM (Out Of Memory).

    Args:
        label (str): Une étiquette pour identifier l'étape du code.
    """
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated(0)
        mem_reserved = torch.cuda.memory_reserved(0)
        print(f"[GPU] {label}: {format_bytes(mem_used)} utilisée / {format_bytes(mem_reserved)} réservée")


def load_images(img_dir):
    """
    Charge toutes les images valides (.jpg, .png) depuis un répertoire donné.
    Convertit automatiquement les images de BGR (OpenCV) vers RGB (attendu par les Transformers).

    Args:
        img_dir (str): Chemin du dossier contenant les images.

    Returns:
        list: Liste de tableaux numpy (images RGB).
    """
    if not os.path.isdir(img_dir):
        return []

    valid_ext = (".jpg", ".jpeg", ".png")
    # Compréhension de liste pour charger et convertir les images
    imgs = [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Conversion BGR -> RGB pour les modèles pré-entraînés
        for fname in os.listdir(img_dir)
        if fname.lower().endswith(valid_ext)
        for img in [cv2.imread(os.path.join(img_dir, fname), cv2.IMREAD_COLOR)]
        if img is not None
    ]
    return imgs


def prepare_input(csv_path, img_base_dir):
    """
    Charge les données tabulaires et associe les images correspondantes.

    Args:
        csv_path (str): Chemin vers le fichier CSV.
        img_base_dir (str): Dossier racine contenant les sous-dossiers d'images (nommés par ID).

    Returns:
        tuple: 
            - dict: {id_annonce (str): liste_images}
            - pd.DataFrame: Le dataframe nettoyé et tronqué (pour le test).
    """
    # Lecture du CSV avec séparateur point-virgule
    dt = pd.read_csv(csv_path, sep=";", quotechar='"')
    # Suppression des colonnes non utilisées pour l'inférence immédiate ou cibles directes
    dt.drop(columns=['images', 'price'], inplace=True)
    # Limitation aux 10 premières lignes pour test/démonstration
    dt = dt[:10]

    # Création du dictionnaire mapping ID -> Liste d'images chargées
    return {str(r['id']): load_images(os.path.join(img_base_dir, str(r['id']))) for _, r in dt.iterrows()}, dt


# ---------------- modules ----------------

class ImageFusionTransformer(nn.Module):
    """
    Module PyTorch utilisant un Transformer Encoder pour fusionner les embeddings 
    de plusieurs images en un seul vecteur global.
    Utilise un token [CLS] apprenable pour agréger l'information.
    """
    def __init__(self, dim, num_layers=2, num_heads=4):
        """
        Initialise le Transformer de fusion.

        Args:
            dim (int): Dimension des vecteurs d'entrée (features image).
            num_layers (int): Nombre de couches TransformerEncoder.
            num_heads (int): Nombre de têtes d'attention.
        """
        super().__init__()
        # Définition d'une couche d'encodage standard
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim*2, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        
        # Token CLS apprenable (1, 1, dim) qui servira de résumé global
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, tokens):
        """
        Passe avant (Forward pass).

        Args:
            tokens (Tensor): Tenseur de forme (Batch, Num_Images, Dim).

        Returns:
            Tensor: Vecteur global fusionné de forme (Batch, Dim).
        """
        B = tokens.size(0)
        # Expansion du token CLS pour correspondre à la taille du batch
        cls = self.cls_token.expand(B, -1, -1)
        # Concaténation : [CLS, Image1, Image2, ...]
        x = torch.cat([cls, tokens], dim=1)
        # Passage dans le Transformer
        out = self.enc(x)
        # On retourne uniquement l'embedding correspondant à la position du CLS (index 0)
        return out[:, 0, :]


class MultiModalMLP(nn.Module):
    """
    Réseau de neurones multicouches (MLP) pour la régression multimodale.
    Fusionne (concatène) les vecteurs Image, Texte et Tabulaire.
    """
    def __init__(self, img_dim, text_dim, tab_dim, hidden=1024):
        """
        Args:
            img_dim (int): Dimension du vecteur image global.
            text_dim (int): Dimension du vecteur texte.
            tab_dim (int): Nombre de features numériques (tabulaires).
            hidden (int): Taille de la couche cachée interne.
        """
        super().__init__()
        # Projections individuelles pour chaque modalité avant fusion
        self.img_fc = nn.Sequential(nn.Linear(img_dim, hidden), nn.ReLU())
        self.text_fc = nn.Sequential(nn.Linear(text_dim, hidden), nn.ReLU())
        self.tab_fc = nn.Sequential(nn.Linear(tab_dim, hidden), nn.ReLU())
        
        # Couche de fusion : entrée = 3 * hidden (car concaténation)
        self.fusion = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU())
        
        # Deux têtes de sortie indépendantes (Multi-task learning)
        self.head1 = nn.Linear(hidden, 1) # Prix vente
        self.head2 = nn.Linear(hidden, 1) # Prix location

    def forward(self, img_vec, text_vec, tab_vec):
        """
        Args:
            img_vec (Tensor): Features images.
            text_vec (Tensor): Features texte.
            tab_vec (Tensor): Features tabulaires brutes.
        """
        # Encodage de chaque modalité
        h_img = self.img_fc(img_vec)
        h_text = self.text_fc(text_vec)
        h_tab = self.tab_fc(tab_vec)
        
        # Concaténation tardive (Late Fusion)
        h = torch.cat([h_img, h_text, h_tab], dim=1)
        
        # Fusion et prédiction
        h = self.fusion(h)
        return self.head1(h), self.head2(h)



def main():
    """
    Fonction principale :
    1. Initialise le système et les arguments.
    2. Charge les modèles (BLIP-2 + Custom modules).
    3. Exécute l'inférence sur les données chargées.
    """
    # --- Gestion des arguments en ligne de commande ---
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv", "--csv_file", type=str, default="../output/csv/annonces_annonces_Auvergne-Rhone-Alpes_0_csv_0.csv")
    parser.add_argument("-i", "--image_dir", type=str, default="../output/images")
    parser.add_argument("--model_id", type=str, default="Salesforce/blip2-flan-t5-xl")
    args = parser.parse_args()

    system_info()
    # Détection automatique du device (GPU Nvidia ou CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # --- Chargement et Préparation des données ---
    data_dict, dt = prepare_input(args.csv_file, args.image_dir)
    
    # Identification des colonnes Textes vs Numériques
    text_cols = [c for c in dt.columns if dt[c].dtype == 'object' and c not in ['id']]
    num_cols = [c for c in dt.columns if dt[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    tab_dim = max(1, len(num_cols)) # Dimension pour le MLP tabulaire

    # Détermination du type de précision (bfloat16 est plus rapide/léger sur Ampere+ GPUs)
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype_load = torch.bfloat16 if use_bf16 else torch.float32
    dtype_for_autocast = torch.bfloat16 if use_bf16 else torch.float32

    print("[INFO] Chargement BLIP-2 (device_map='auto')...")
    # Chargement du modèle BLIP-2 pré-entraîné
    # device_map="auto" répartit automatiquement les couches sur le GPU/CPU si manque de VRAM
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_id, device_map="auto",
        dtype=dtype_load)

    model.eval() # Mise en mode évaluation (désactive dropout, batchnorm update, etc.)
    processor = Blip2Processor.from_pretrained(args.model_id, use_fast=True)

    # --- Gel (Freezing) des poids du modèle BLIP-2 ---
    # On utilise BLIP-2 uniquement comme extracteur de caractéristiques (Feature Extractor),
    # on ne souhaite pas ré-entraîner ses poids (Backpropagation désactivée).
    for p in model.vision_model.parameters():
        p.requires_grad = False
    for p in model.qformer.parameters():
        p.requires_grad = False
    
    # Gestion spécifique des couches projecteurs si elles existent
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
    # Passage d'une image "dummy" pour récupérer dynamiquement la taille de sortie visuelle
    dummy_pixel = torch.zeros(1, 3, 224, 224).to(device)
    img_feat_model_dim = model.get_image_features(dummy_pixel).shape[-1]

    img_feat_dim = 768  # Dimension cible arbitraire pour la fusion/MLP
    # Récupération de la dimension texte (souvent identique à la vision dans BLIP, mais prudence)
    text_feat_dim = model.text_projection.out_features if hasattr(model, "text_projection") else img_feat_dim

    # Couche linéaire pour projeter la sortie de BLIP vers la dimension commune (768)
    img_proj = nn.Linear(img_feat_model_dim, img_feat_dim).to(device).eval()

    # Initialisation des modules "downstream" (Fusion + MLP) et envoi sur GPU
    fusion_transformer = ImageFusionTransformer(img_feat_dim).to(device).eval()
    multimodal_mlp = MultiModalMLP(img_feat_dim, text_feat_dim, tab_dim, hidden=512).to(device).eval()

    print(f"[INFO] dims img:{img_feat_dim} text:{text_feat_dim} tab:{tab_dim}")
    gpu_memory("après init")
    print(f"[INFO] using autocast dtype: {dtype_for_autocast}")

    # --- Boucle d'inférence ---
    for _, row in dt.iterrows():
        id_str = str(row['id'])
        images = data_dict.get(id_str, [])
        if len(images) == 0:
            continue # On ignore s'il n'y a pas d'images

        # --- Prépare images ---
        # Le processeur normalise, redimensionne et créé les tenseurs
        img_inputs = processor(images=images, return_tensors="pt", padding=True)
        pixel_values = img_inputs['pixel_values'].to(device)

        # --- Prépare Texte ---
        # Concaténation de toutes les colonnes textuelles
        text_data = " | ".join([str(row.get(c, "")) for c in text_cols if c in row.index]) if len(text_cols) > 0 else ""
        # Tokenisation
        text_inputs = processor(text=text_data, return_tensors="pt", padding=True, truncation=True)

        # --- Features numériques ---
        # Extraction et conversion en Tensor
        num_vals = [float(row.get(c, 0.0)) for c in num_cols] if num_cols else [0.0]*tab_dim
        num_feats = torch.tensor(num_vals, dtype=torch.float32, device=device).unsqueeze(0)

        # --- Extraction features et Passage Forward ---
        with torch.no_grad(): # Désactive le calcul des gradients (économie mémoire + vitesse)

            # Traitement Vision (potentiellement lourd, d'où l'autocast)
            with autocast(device_type='cuda', dtype=dtype_for_autocast):
                # 1. Extraction features brutes via BLIP Vision Model
                image_feats = model.get_image_features(pixel_values=pixel_values)
                # 2. Moyenne simple par image (si le modèle renvoie plusieurs patchs par image)
                per_image_vecs = image_feats.mean(dim=1)
                # 3. Projection vers dimension 768 et ajout dimension Batch (1, N_images, Dim)
                tokens_for_fusion = img_proj(per_image_vecs).unsqueeze(0)
                # 4. Fusion des N images via le Transformer custom
                img_global = fusion_transformer(tokens_for_fusion).to(dtype=torch.float32)

            # Traitement Texte
            try:
                # Extraction embedding sémantique du texte
                text_feats = model.get_text_features(**text_inputs)
                text_embeds = text_feats.to(device=device, dtype=torch.float32)
            except Exception:
                # Fallback si erreur (texte vide ou bug modèle)
                text_embeds = torch.zeros((1, text_feat_dim), device=device, dtype=torch.float32)

            # Forward downstream : Concaténation (Img + Txt + Tab) -> MLP -> Sorties
            out1, out2 = multimodal_mlp(img_global, text_embeds, num_feats)

        # Vérification de sécurité (NaN)
        if torch.isnan(out1).any() or torch.isnan(out2).any():
            print(f"[ERROR] NaN detected for id={id_str}")
            continue

        print(f"[OUTPUT] id={id_str} Prix vente: {float(out1.item()):.2f} | Prix location: {float(out2.item()):.2f}")


if __name__ == '__main__':
    main()
