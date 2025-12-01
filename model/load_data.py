import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from PIL import Image



# Mappings Ordinal (Texte -> Chiffre)
DPE_MAP = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2}
ETAT_MAP = {
    'neuf': 5, 'refait a neuf': 5, 'excellent': 5,
    'tres bon': 4, 'bon': 3,
    'correct': 2, 'usage': 2,
    'travaux': 1, 'renover': 0
}

# À rafraichir; Bon état; Rénové; Travaux à prévoir; Très bon état


def get_cols_config():
    """
    Retourne la configuration SOTA des colonnes basée sur l'expertise du dataset.
    """

    # 1. TEXTE (Concaténation pour CamemBERT)
    # Ces colonnes seront fusionnées : "Titre . Specificités . Description"
    text_cols = ['titre', 'specificites', 'description']

    # 2. CATÉGORIES PURES (Embeddings Classiques)
    # Variables nominales sans ordre mathématique logique.
    cat_cols = [
        'type_bien',  # Maison vs Appartement : pas d'ordre
        'orientation', # Nord vs Sud : pas d'ordre linéaire simple
        'exterior_access'
    ]

    # 3. CONTINUS (Periodic Embeddings)
    # Regroupe :
    # - Les vrais continus (Surface, GPS)
    # - Les compteurs (Pièces, Chambres)
    # - Les ordinaux convertis (DPE, État) -> Le modèle apprendra la courbe de prix exacte.
    cont_cols = [
        'latitude', 
        'longitude', 
        'surface_habitable', 
        'surface_tolale_terrain', 
        'nb_pieces', 
        'nb_chambres', 
        'nb_salleDeBains', 
        'nb_placesParking', 
        'nb_etages_Immeuble', 
        'annee_construction', 
        'classe_energetique', # SOTA: Traité comme un score (1 à 8)
        'etat_bien'           # SOTA: Traité comme un score (0 à 5)
    ]

    # Note : 'id', 'prix', 'images_urls' sont ignorés ici car gérés spécifiquement
    # dans le Dataset (Meta-données ou Cibles).

    return cont_cols, cat_cols, text_cols


def prepare_preprocessors(csv_folder, cont_cols, cat_cols):
    """
    Calibre le Scaler (pour les chiffres) et les Index (pour les catégories).

    Stratégie SOTA :
    - Continus : StandardScaler (Moyenne=0, Variance=1). Indispensable pour que les 
      PeriodicEmbeddings et le réseau convergent vite.
    - Catégories : On crée un dictionnaire unique pour transformer le texte en ID.
    """
    print("[PREPROC] Calibration des données (Normalisation & Indexation)...")

    df_list = []
    # On liste les 5er fichiers CSV
    files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder)][:5]

    for f in files: 
        try:
            df = pd.read_csv(f)

            # --- CONVERSIONS PRÉALABLES INDISPENSABLES ---
            # Le Scaler ne sait lire que des chiffres. Il faut convertir DPE/ETAT maintenant.

            # 1. DPE (Lettre -> Chiffre)
            df['classe_energetique'] = df['classe_energetique'].astype(str).str.upper().map(DPE_MAP)

            # 2. ETAT (Texte -> Chiffre)
            df['etat_bien'] = df['etat_bien'].astype(str).map(ETAT_MAP)

            df_list.append(df)
        except Exception as e:
            print(e)

    full_df = pd.concat(df_list, ignore_index=True)

    # ==========================================================================
    # 1. CALIBRATION CONTINUE (SCALER)
    # ==========================================================================

    scaler = StandardScaler()
    scaler.fit(full_df[cont_cols].values)

    # ==========================================================================
    # 2. CALIBRATION CATÉGORIELLE (INDEXATION)
    # ==========================================================================
    cat_mappings = {}
    cat_dims = []

    for c in cat_cols:
        # On convertit tout en string
        full_df[c] = full_df[c].astype(str)
  
        # On récupère les valeurs uniques
        uniques = sorted(full_df[c].unique())

        # Création du dictionnaire : Valeur -> ID
        mapping = {val: i for i, val in enumerate(uniques)}

        # Ajout du token <UNK> pour la robustesse en production
        mapping["<UNK>"] = len(uniques)
 
        cat_mappings[c] = mapping
        cat_dims.append(len(mapping) + 1)

    print(f"[PREPROC] Terminé. Scaler calibré sur {len(cont_cols)} variables.")

    # On retourne None pour les médianes car on suppose que vous gérez les NaN ailleurs
    return scaler, None, cat_mappings, cat_dims


class RealEstateDataset(Dataset):
    def __init__(self, csv_folder, img_dir, tokenizer, scaler, medians, cat_mappings, cont_cols, cat_cols, is_train=True):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.medians = medians
        self.cat_mappings = cat_mappings
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        
        # Chargement
        self.df = pd.concat([pd.read_csv(os.path.join(csv_folder, f)) for f in os.listdir(csv_folder) if f.endswith('.csv')], ignore_index=True)

        # Nettoyage Targets
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
        self.df = self.df.dropna(subset=['price'])
        self.df['log_price'] = np.log1p(self.df['price'])
        # Placeholder loyer (à adapter si vous l'avez)
        self.df['log_rent'] = np.zeros(len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. TEXTE (Fusion Titre + Specificites + Description)
        # C'est ici qu'on gère les spécificités via BERT
        parts = [
            str(row.get('titre', '')),
            str(row.get('specificites', '')),
            str(row.get('description', ''))
        ]
        full_text = " . ".join([p for p in parts if p and str(p) != 'nan'])
        
        tokens = self.tokenizer(full_text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

        # 2. CONTINUS (Avec Mapping DPE/Etat à la volée)
        vals = []
        for c in self.cont_cols:
            val = row.get(c, np.nan)
            
            # Mapping DPE
            if c == 'classe_energetique':
                val = DPE_MAP.get(str(val).upper(), 4.0)
            # Mapping Etat
            elif c == 'etat_bien':
                val_str = str(val).lower()
                val = 3.0 # Default
                for k, v in ETAT_MAP.items():
                    if k in val_str: 
                        val = float(v)
                        break
            
            # Fallback numérique
            try:
                val = float(val)
                if np.isnan(val): val = self.medians.get(c, 0)
            except:
                val = self.medians.get(c, 0)
            
            vals.append(val)
            
        raw_cont = np.array(vals, dtype=float).reshape(1, -1)
        scaled_cont = self.scaler.transform(raw_cont).flatten()

        # 3. CATEGORIES
        cat_idxs = []
        for c in self.cat_cols:
            val = str(row.get(c, "MISSING"))
            cat_idxs.append(self.cat_mappings[c].get(val, self.cat_mappings[c]["<UNK>"]))

        # 4. IMAGES (Via URL ou Path Local)
        # TODO: Implémentez votre logique de chargement d'image ici
        # img_path = row['images_urls']... download or open local
        img_tensor = torch.zeros((1, 3, 224, 224)) # Dummy placeholder

        return {
            'images': img_tensor,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'tab_cont': torch.tensor(scaled_cont, dtype=torch.float32),
            'tab_cat': torch.tensor(cat_idxs, dtype=torch.long),
            'targets': torch.tensor([row['log_price'], 0.0], dtype=torch.float32),
            'masks': torch.tensor([1.0, 0.0], dtype=torch.float32)
        }

def real_estate_collate_fn(batch):
    return {
        'images': torch.stack([x['images'] for x in batch]),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'tab_cont': torch.stack([x['tab_cont'] for x in batch]),
        'tab_cat': torch.stack([x['tab_cat'] for x in batch]),
        'targets': torch.stack([x['targets'] for x in batch]),
        'masks': torch.stack([x['masks'] for x in batch])
    }
