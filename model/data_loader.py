"""
load_data.py - Chargeur Multi-Task (Vente & Location).

Corrections :
- dataset_source est utilisé comme Feature d'entrée (0=Vente, 1=Loc).
- dataset_source est utilisé pour router la Target (remplir log_price OU log_rent).
"""

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torchvision import transforms



# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

BINARY_COLS = [
    'ext_balcony', 'ext_garden', 'ext_pool', 'ext_terrace', 
    'feat_american_kitchen', 'feat_attic', 'feat_basement', 'feat_caretaker', 'feat_cellar',
    'feat_equipped_kitchen', 'feat_heated_floor', 'feat_historical_building', 'feat_impaired_mobility_friendly',
    'feat_intercom', 'feat_new_building', 'feat_old_builiding', 'feat_with_dependency',
    'feat_with_garage_or_parking_spot'
]


def get_cols_config():
    # 1. Texte
    text_cols = ['titre', 'description']

    # 2. Catégories
    cat_cols = ['property_type', 'orientation']

    # 3. Continus
    base_cont = [
        'latitude', 'longitude', 
        'living_area_sqm', 'total_land_area_sqm', 
        'num_rooms', 'num_bedrooms', 'num_bathrooms', 'num_parking_spaces', 'num_floors',
        'year_built', 
        'energy_rating', 
        'property_status',
        'dataset_source' # <--- Le modèle doit savoir le type de transaction
    ]

    cont_cols = base_cont + BINARY_COLS
    return cont_cols, cat_cols, text_cols


# ==============================================================================
# 2. CALIBRATION
# ==============================================================================
def prepare_preprocessors(csv_folder, cont_cols, cat_cols):
    print(f"[PREPROC] Calibration sur {csv_folder}...")
    
    df_list = []
    try:
        files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder)][:5]
    except: return None, None, None, None

    for f in files: 
        try:
            df = pd.read_csv(f, sep=";", quotechar='"')
            df_list.append(df)
        except: continue

    if not df_list: raise ValueError("Aucun CSV valide.")
    full_df = pd.concat(df_list, ignore_index=True)

    # --- SCALER ---
    # On exclut dataset_source du scaling car c'est déjà 0/1 (binaire pur)
    # Mais le StandardScaler gère bien les 0/1, donc on peut tout scaler pour simplifier.
    valid_cont = [c for c in cont_cols if c in full_df.columns]
    
    medians = full_df[valid_cont].median().to_dict()
    full_df[valid_cont] = full_df[valid_cont].fillna(medians)

    scaler = StandardScaler()
    scaler.fit(full_df[valid_cont].values)

    # --- MAPPINGS ---
    cat_mappings = {}
    cat_dims = []
    for c in cat_cols:
        full_df[c] = full_df[c].astype(str).fillna("MISSING")
        uniques = sorted(full_df[c].unique())
        mapping = {val: i for i, val in enumerate(uniques)}
        mapping["<UNK>"] = len(uniques)
        cat_mappings[c] = mapping
        cat_dims.append(len(mapping) + 1)

    return scaler, medians, cat_mappings, cat_dims



# ==============================================================================
# 3. DATASET MULTI-TASK
# ==============================================================================
class RealEstateDataset(Dataset):
    def __init__(self, csv_folder, img_dir, tokenizer, scaler, medians, cat_mappings, cont_cols, cat_cols, is_train=True):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.medians = medians
        self.cont_cols = cont_cols
        self.cat_mappings = cat_mappings
        self.cat_cols = cat_cols

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Chargement
        self.data = []
        files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
        for f in files:
            try:
                self.data.append(pd.read_csv(os.path.join(csv_folder, f), sep=None, engine='python'))
            except: pass
        self.df = pd.concat(self.data, ignore_index=True)

        # --- GESTION DES TARGETS (VENTE vs LOCATION) ---
        # Nettoyage prix brut
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
        self.df = self.df.dropna(subset=['price'])

        # Nettoyage Source (0 = Vente, 1 = Location, ajustez selon vos données réelles !)
        # On suppose ici que dans votre CSV : 0 = Vente, 1 = Location.
        # Si c'est du texte ("buy", "rent"), il faut le convertir avant.
        self.df['dataset_source'] = pd.to_numeric(self.df['dataset_source'], errors='coerce').fillna(0).astype(int)


    def __len__(self):
        return len(self.df)


    def load_images(self, property_id):
        clean_id = str(int(float(property_id))) if pd.notna(property_id) else "unknown"
        folder_path = os.path.join(self.img_dir, clean_id)
        images = []
        if os.path.isdir(folder_path):
            files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))])
            for f in files[:5]:
                try:
                    with Image.open(os.path.join(folder_path, f)).convert('RGB') as img:
                        images.append(self.transform(img))
                except: continue
        if not images: images.append(torch.zeros(3, 224, 224))
        return torch.stack(images)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. TEXTE
        text = str(row.get('titre', '')) + " . " + str(row.get('description', ''))
        tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

        # 2. CONTINUS (Inclus dataset_source maintenant !)
        vals = []
        for c in self.cont_cols:
            val = row.get(c, np.nan)
            vals.append(val if pd.notna(val) else self.medians.get(c, 0.0))
        scaled_cont = self.scaler.transform(np.array(vals).reshape(1, -1)).flatten()

        # 3. CATÉGORIES
        cat_idxs = []
        for c in self.cat_cols:
            val = str(row.get(c, "MISSING"))
            cat_idxs.append(self.cat_mappings[c].get(val, self.cat_mappings[c]["<UNK>"]))

        # 4. TARGETS & MASQUES (LOGIQUE MULTI-TASK)
        source = int(row.get('dataset_source', 0))
        price_val = float(row['price'])
        log_val = np.log1p(price_val)

        # Initialisation : Tout à zéro
        target_vec = torch.zeros(2, dtype=torch.float32) # [Log_Vente, Log_Loc]
        mask_vec = torch.zeros(2, dtype=torch.float32)   # [Mask_Vente, Mask_Loc]

        if source == 0: 
            # C'est une VENTE
            target_vec[0] = log_val
            mask_vec[0] = 1.0
        else:
            # C'est une LOCATION
            target_vec[1] = log_val
            mask_vec[1] = 1.0

        # 5. IMAGES
        images_tensor = self.load_images(row['id'])

        return {
            'images': images_tensor,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'tab_cont': torch.tensor(scaled_cont, dtype=torch.float32),
            'tab_cat': torch.tensor(cat_idxs, dtype=torch.long), # Toujours nécessaire pour property_type/orientation
            'targets': target_vec,
            'masks': mask_vec
        }


# --- COLLATE ---
def real_estate_collate_fn(batch):
    max_imgs = max([x['images'].shape[0] for x in batch])
    padded_images = []
    for x in batch:
        imgs = x['images']
        if imgs.shape[0] < max_imgs:
            pad = torch.zeros((max_imgs - imgs.shape[0], 3, 224, 224))
            imgs = torch.cat([imgs, pad], dim=0)
        padded_images.append(imgs)
        
    return {
        'images': torch.stack(padded_images),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'tab_cont': torch.stack([x['tab_cont'] for x in batch]),
        'tab_cat': torch.stack([x['tab_cat'] for x in batch]),
        'targets': torch.stack([x['targets'] for x in batch]),
        'masks': torch.stack([x['masks'] for x in batch])
    }
