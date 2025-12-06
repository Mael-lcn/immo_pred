import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
from torchvision import transforms



# ==============================================================================
# CONFIGURATION
# ==============================================================================

# On déplace les binaires ici pour qu'elles soient traitées comme des EMBEDDINGS
BINARY_COLS = [
    'ext_balcony', 'ext_garden', 'ext_pool', 'ext_terrace', 
    'feat_american_kitchen', 'feat_attic', 'feat_basement', 'feat_caretaker', 'feat_cellar',
    'feat_equipped_kitchen', 'feat_heated_floor', 'feat_historical_building', 'feat_impaired_mobility_friendly',
    'feat_intercom', 'feat_new_building', 'feat_old_builiding', 'feat_with_dependency',
    'feat_with_garage_or_parking_spot'
]


def get_cols_config():
    text_cols = ['titre', 'description']

    # Ajout des binaires aux catégories + dataset_source
    base_cat = ['property_type', 'orientation', 'dataset_source']
    cat_cols = base_cat + BINARY_COLS

    # Uniquement les vrais chiffres continus ici
    cont_cols = [
        'latitude', 'longitude', 
        'living_area_sqm', 'total_land_area_sqm', 
        'num_rooms', 'num_bedrooms', 'num_bathrooms', 'num_parking_spaces',
        'year_built', 'energy_rating'
    ]
    
    return cont_cols, cat_cols, text_cols


# ==============================================================================
# CALIBRATION
# ==============================================================================
def prepare_preprocessors(csv_folder, cont_cols, cat_cols):
    print(f"[PREPROC] Calibration sur {csv_folder}...")
    df_list = []
    try:
        files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith('.csv')][:6]
    except: return None, None, None, None

    for f in files: 
        try: df_list.append(pd.read_csv(f, sep=None, engine='python'))
        except: continue

    if not df_list: raise ValueError("Aucun CSV valide.")
    full_df = pd.concat(df_list, ignore_index=True)

    # 1. Gestion des valeurs manquantes et Log Transform pour les surfaces
    medians = {}
    for c in cont_cols:
        full_df[c] = pd.to_numeric(full_df[c], errors='coerce')
        med = full_df[c].median()
        medians[c] = med
        full_df[c] = full_df[c].fillna(med)
        
        # Astuce SOTA : Log-transform sur les surfaces pour écraser les outliers
        if 'area' in c:
            full_df[c] = np.log1p(full_df[c])

    # 2. Scaler Robuste (Gère mieux les outliers que StandardScaler)
    scaler = RobustScaler()
    scaler.fit(full_df[cont_cols].values)

    # 3. Mappings Catégoriels (Y compris pour les binaires 0/1)
    cat_mappings = {}
    cat_dims = []

    for c in cat_cols:
        # On force en string pour gérer "0.0", "1.0", "True", "False" uniformément
        full_df[c] = full_df[c].astype(str).fillna("MISSING")
        
        # Nettoyage spécifique pour les binaires (0.0 -> 0)
        if c in BINARY_COLS or c == 'dataset_source':
             full_df[c] = full_df[c].apply(lambda x: str(int(float(x))) if x.replace('.','',1).isdigit() else "0")

        uniques = sorted(full_df[c].unique())
        mapping = {val: i for i, val in enumerate(uniques)}
        mapping["<UNK>"] = len(uniques) # Token inconnu
        
        cat_mappings[c] = mapping
        cat_dims.append(len(mapping) + 1)

    return scaler, medians, cat_mappings, cat_dims


# ==============================================================================
# DATASET
# ==============================================================================
class RealEstateDataset(Dataset):
    def __init__(self, csv_folder, img_dir, tokenizer, scaler, medians, cat_mappings, cont_cols, cat_cols):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.medians = medians
        self.cont_cols = cont_cols
        self.cat_mappings = cat_mappings
        self.cat_cols = cat_cols
        self.binary_cols = BINARY_COLS # Référence globale

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Normalisation ImageNet standard
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.data = []
        files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
        for f in files:
            try: self.data.append(pd.read_csv(os.path.join(csv_folder, f), sep=None, engine='python'))
            except: pass
        self.df = pd.concat(self.data, ignore_index=True)
        
        # Nettoyage Prix
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
        self.df = self.df.dropna(subset=['price'])
        # Nettoyage Source (0/1)
        self.df['dataset_source'] = pd.to_numeric(self.df['dataset_source'], errors='coerce').fillna(0).astype(int)


    def __len__(self):
        return len(self.df)


    def load_images(self, property_id):
        folder_path = os.path.join(self.img_dir, str(property_id))
        images = []
        if os.path.exists(folder_path):
            files = sorted(os.listdir(folder_path))[:10] # Max 10 images
            for f in files:
                try:
                    with Image.open(os.path.join(folder_path, f)).convert('RGB') as img:
                        images.append(self.transform(img))
                except: continue
        
        if not images:
            # Image noire "placeholder" si vide
            images.append(torch.zeros(3, 224, 224))
            return torch.stack(images), False # False = C'est du fake
        
        return torch.stack(images), True # True = Vraies images


    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Texte
        text = str(row.get('titre', '')) + " . " + str(row.get('description', ''))
        tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

        # 2. Continus (Log1p + RobustScaler)
        vals = []
        for c in self.cont_cols:
            raw_val = pd.to_numeric(row.get(c, np.nan), errors='coerce')
            if pd.isna(raw_val): raw_val = self.medians.get(c, 0.0)

            # Application Log1p si c'est une surface
            if 'area' in c: raw_val = np.log1p(max(0, raw_val))
            vals.append(raw_val)

        scaled_cont = self.scaler.transform(np.array(vals).reshape(1, -1)).flatten()

        # 3. Catégories (y compris Binaires)
        cat_idxs = []
        for c in self.cat_cols:
            val_raw = str(row.get(c, "MISSING"))
            # Normalisation string comme dans le prepare_preprocessors
            if c in self.binary_cols or c == 'dataset_source':
                try: val_clean = str(int(float(val_raw)))
                except: val_clean = "0"
            else:
                val_clean = val_raw
                
            idx_cat = self.cat_mappings[c].get(val_clean, self.cat_mappings[c]["<UNK>"])
            cat_idxs.append(idx_cat)

        # 4. Target
        source = int(row.get('dataset_source', 0))
        log_price = np.log1p(float(row['price']))
        target_vec = torch.zeros(2); mask_vec = torch.zeros(2)
        
        if source == 0: target_vec[0] = log_price; mask_vec[0] = 1.0 # Vente
        else: target_vec[1] = log_price; mask_vec[1] = 1.0 # Loc

        # 5. Images
        imgs_tensor, has_real_imgs = self.load_images(row['id'])

        return {
            'images': imgs_tensor,
            'has_real_imgs': has_real_imgs, # Pour savoir si le dossier était vide
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'tab_cont': torch.tensor(scaled_cont, dtype=torch.float32),
            'tab_cat': torch.tensor(cat_idxs, dtype=torch.long),
            'targets': target_vec,
            'masks': mask_vec
        }


def real_estate_collate_fn(batch):
    img_lengths = [x['images'].shape[0] for x in batch]
    max_imgs = max(img_lengths)
    batch_size = len(batch)

    padded_images = torch.zeros(batch_size, max_imgs, 3, 224, 224)
    # Masque d'attention pour les images (1 = Vraie Image, 0 = Padding)
    image_attn_mask = torch.zeros(batch_size, max_imgs)

    for i, x in enumerate(batch):
        n = x['images'].shape[0]
        padded_images[i, :n] = x['images']

        # Si le dossier était vide (has_real_imgs=False), on met 0 partout (même l'image noire est ignorée)
        # Sinon, on met 1 sur les n images présentes
        if x['has_real_imgs']:
            image_attn_mask[i, :n] = 1.0
        else:
            # Cas rare dossier vide : on laisse tout à 0, le modèle ignorera totalement la vision
            pass 

    return {
        'images': padded_images,
        'image_masks': image_attn_mask,
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'tab_cont': torch.stack([x['tab_cont'] for x in batch]),
        'tab_cat': torch.stack([x['tab_cat'] for x in batch]),
        'targets': torch.stack([x['targets'] for x in batch]),
        'masks': torch.stack([x['masks'] for x in batch])
    }
