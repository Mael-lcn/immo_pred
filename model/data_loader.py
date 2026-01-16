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

BINARY_COLS = [
    'ext_balcony', 'ext_garden', 'ext_pool', 'ext_terrace', 
    'feat_american_kitchen', 'feat_attic', 'feat_basement', 'feat_caretaker', 'feat_cellar',
    'feat_equipped_kitchen', 'feat_heated_floor', 'feat_historical_building', 'feat_impaired_mobility_friendly',
    'feat_intercom', 'feat_new_building', 'feat_old_builiding', 'feat_with_dependency',
    'feat_with_garage_or_parking_spot'
]

def get_cols_config():
    text_cols = ['titre', 'description']
    base_cat = ['property_type', 'orientation', 'dataset_source']
    cat_cols = base_cat + BINARY_COLS

    # Variables purement continues
    cont_cols = [
        'latitude', 'longitude', 
        'living_area_sqm', 'total_land_area_sqm', 
        'num_rooms', 'num_bedrooms', 'num_bathrooms', 'num_parking_spaces',
        'year_built', 'energy_rating'
    ]
    return cont_cols, cat_cols, text_cols

# ==============================================================================
# CALIBRATION (Fit sur le Train uniquement !)
# ==============================================================================
def prepare_preprocessors(csv_path_or_df, cont_cols, cat_cols):
    """
    Calcule les stats (Médiane, Mode, Scaler, Vocabulaire) sur le jeu d'entraînement.
    """
    print(f"[PREPROC] Calibration en cours...")

    if isinstance(csv_path_or_df, str):
        # Si c'est un dossier
        df_list = []
        try:
            files = [os.path.join(csv_path_or_df, f) for f in os.listdir(csv_path_or_df) if f.endswith('.csv')]
            # On prend tout le dossier pour la calibration (ou juste le train set si séparé avant)
            for f in files: 
                try: df_list.append(pd.read_csv(f, sep=None, engine='python'))
                except: continue
            full_df = pd.concat(df_list, ignore_index=True)
        except Exception as e:
            raise ValueError(f"Erreur lecture CSV: {e}")
    else:
        # Si c'est déjà un DataFrame
        full_df = csv_path_or_df.copy()

    # --- 1. Gestion Variables Continues (Médiane + Log + Scaler) ---
    medians = {}

    # Conversion préventive en numérique
    for c in cont_cols:
        full_df[c] = pd.to_numeric(full_df[c], errors='coerce')
        
        # Calcul de la médiane (sur les valeurs non-nulles)
        med = full_df[c].median()
        if pd.isna(med): med = 0.0 # Fallback si colonne vide
        medians[c] = med

        # Remplissage des NaNs
        full_df[c] = full_df[c].fillna(med)

        """
        # Log Transform pour les surfaces (Lissage des distributions exponentielles) add long et lat ou rien faire
        if 'area' in c:
            full_df[c] = np.log1p(full_df[c])
        """

    # Fit du Scaler Robuste
    scaler = RobustScaler()
    scaler.fit(full_df[cont_cols].values)

    # --- 2. Gestion Variables Catégorielles (Mode + Vocabulaire) ---
    cat_mappings = {}
    cat_dims = []
    modes = {}

    for c in cat_cols:
        # Conversion string pour homogénéiser
        s_col = full_df[c].astype(str)

        # Nettoyage spécifique pour Binaires/Source ("1.0" -> "1")
        if c in BINARY_COLS or c == 'dataset_source':
             s_col = s_col.apply(lambda x: str(int(float(x))) if x.replace('.','',1).isdigit() else "NaN")

        # Calcul du MODE (Valeur la plus fréquente)
        # On exclut "nan", "NaN", "MISSING" du calcul du mode
        valid_vals = s_col[~s_col.isin(["nan", "NaN", "MISSING", "None"])]
        if len(valid_vals) > 0:
            mode_val = valid_vals.mode()[0]
        else:
            mode_val = "0" if c in BINARY_COLS else "MISSING" # Fallback

        modes[c] = mode_val

        # On remplit les NaNs dans le DF de calibration pour construire le vocabulaire propre
        s_col = s_col.replace(["nan", "NaN", "MISSING", "None"], mode_val)

        # Création du vocabulaire de type mot1: 1
        uniques = sorted(s_col.unique())
        mapping = {val: i for i, val in enumerate(uniques)}

        # Token <UNK> pour les valeurs jamais vues
        mapping["<UNK>"] = len(uniques)

        cat_mappings[c] = mapping
        cat_dims.append(len(mapping) + 1) # +1 pour <UNK>

    print(f"[PREPROC] Terminé. {len(cont_cols)} vars continues, {len(cat_cols)} vars catégorielles.")
    return scaler, medians, modes, cat_mappings, cat_dims


# ==============================================================================
# DATASET
# ==============================================================================
class RealEstateDataset(Dataset):
    def __init__(self, df_or_folder, img_dir, tokenizer, scaler, medians, modes, cat_mappings, cont_cols, cat_cols):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.medians = medians
        self.modes = modes # Dictionnaire des modes
        self.cat_mappings = cat_mappings
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        self.binary_cols = BINARY_COLS

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Chargement des données
        if isinstance(df_or_folder, pd.DataFrame):
            self.df = df_or_folder.copy()
        else:
            data = []
            files = [f for f in os.listdir(df_or_folder) if f.endswith('.csv')]
            for f in files:
                try: data.append(pd.read_csv(os.path.join(df_or_folder, f), sep=None, engine='python'))
                except: pass
            self.df = pd.concat(data, ignore_index=True)

        # Nettoyage Targets
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
        self.df = self.df.dropna(subset=['price'])
        self.df['dataset_source'] = pd.to_numeric(self.df['dataset_source'], errors='coerce').fillna(0).astype(int)


    def __len__(self):
        return len(self.df)


    def load_images(self, property_id):
        folder_path = os.path.join(self.img_dir, str(property_id))
        images = []
        if os.path.exists(folder_path):
            # Filtrage des extensions valides uniquement
            valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            files = sorted([f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts])[:10]

            for f in files:
                try:
                    with Image.open(os.path.join(folder_path, f)).convert('RGB') as img:
                        images.append(self.transform(img))
                except: continue

        if not images:
            # Placeholder: Image noire (C, H, W)
            # IMPORTANT : On renvoie False pour dire "C'est du fake"
            images.append(torch.zeros(3, 224, 224))
            return torch.stack(images), False 

        return torch.stack(images), True


    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- 1. Texte ---
        titre = str(row.get('titre', ''))
        desc = str(row.get('description', ''))
        if titre == 'nan': titre = ""
        if desc == 'nan': desc = ""
        text = (titre + " . " + desc).strip()
        if not text: text = "bien immobilier" # Fallback texte vide

        tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

        # --- 2. Continus (Imputation Mediane + Log1p + Scaler) ---
        vals = []
        for c in self.cont_cols:
            raw_val = pd.to_numeric(row.get(c, np.nan), errors='coerce')

            # IMPUTATION PAR LA MÉDIANE
            if pd.isna(raw_val): 
                raw_val = self.medians.get(c, 0.0)
            """
            # Log1p pour surface mettre lat et long ou laisse commenté
            if 'area' in c: raw_val = np.log1p(max(0, raw_val))
            vals.append(raw_val)
            """

        # Transformation via le Scaler ajusté précédemment
        scaled_cont = self.scaler.transform(np.array(vals).reshape(1, -1)).flatten()

        # --- 3. Catégories (Imputation Mode + Mapping) ---
        cat_idxs = []
        for c in self.cat_cols:
            val_raw = str(row.get(c, "nan"))

            # Nettoyage Binaire/Source
            if c in self.binary_cols or c == 'dataset_source':
                try: 
                    # Tente de convertir float->int->str (ex: "1.0" -> "1")
                    val_clean = str(int(float(val_raw)))
                except: 
                    # Si échec (ex: "nan"), on met le placeholder temporaire
                    val_clean = "nan"
            else:
                val_clean = val_raw

            # IMPUTATION PAR LE MODE
            if val_clean in ["nan", "NaN", "MISSING", "None", ""]:
                val_clean = self.modes.get(c, "MISSING") # Utilise le mode calculé au fit

            # Récupération de l'index (ou <UNK> si valeur jamais vue au train)
            idx_cat = self.cat_mappings[c].get(val_clean, self.cat_mappings[c]["<UNK>"])
            cat_idxs.append(idx_cat)

        # --- 4. Target ---
        source = int(row.get('dataset_source', 0))
        # Log du prix pour stabiliser la regression
        log_price = np.log1p(float(row['price']))

        # Vecteur cible [Vente, Location]
        target_vec = torch.zeros(2)
        mask_vec = torch.zeros(2) # 1 si la target existe, 0 sinon

        if source == 0: # Vente
            target_vec[0] = log_price
            mask_vec[0] = 1.0 
        else: # Location
            target_vec[1] = log_price
            mask_vec[1] = 1.0

        # --- 5. Images ---
        imgs_tensor, has_real_imgs = self.load_images(row['id'])

        return {
            'images': imgs_tensor,         # (N, 3, 224, 224)
            'has_real_imgs': has_real_imgs, # Bool
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0), # Masque texte
            'tab_cont': torch.tensor(scaled_cont, dtype=torch.float32),
            'tab_cat': torch.tensor(cat_idxs, dtype=torch.long),
            'targets': target_vec,
            'masks': mask_vec
        }


def real_estate_collate_fn(batch):
    """
    Gère le padding dynamique des images dans le batch.
    """
    img_lengths = [x['images'].shape[0] for x in batch]
    max_imgs = max(img_lengths)
    batch_size = len(batch)

    # Tenseur global d'images (B, Max_N, 3, H, W)
    padded_images = torch.zeros(batch_size, max_imgs, 3, 224, 224)

    # Masque d'attention Image : 1 = Vrai pixel, 0 = Padding/Fake
    # Le modèle fera (image_masks == 0) -> True pour ignorer
    image_attn_mask = torch.zeros(batch_size, max_imgs)

    for i, x in enumerate(batch):
        n = x['images'].shape[0]
        padded_images[i, :n] = x['images']

        if x['has_real_imgs']:
            # On marque les N images comme valides
            image_attn_mask[i, :n] = 1.0
        else:
            # Dossier vide : has_real_imgs est False.
            # On laisse image_attn_mask à 0 partout.
            # L'attention du modèle ignorera TOUT le bloc image pour cet item.
            pass 

    return {
        'images': padded_images,
        'image_masks': image_attn_mask,
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'text_mask': torch.stack([x['attention_mask'] for x in batch]),
        'x_cont': torch.stack([x['tab_cont'] for x in batch]),
        'x_cat': torch.stack([x['tab_cat'] for x in batch]),
        'targets': torch.stack([x['targets'] for x in batch]),
        'masks': torch.stack([x['masks'] for x in batch])
    }
