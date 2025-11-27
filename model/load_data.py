import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd
import math
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler



# --- FONCTIONS UTILITAIRES ---
def get_cols_config(csv_path, sep=";"):
    """Lit la première ligne pour détecter les colonnes."""
    df_sample = pd.read_csv(csv_path, sep=sep, nrows=5)
    # On ignore les colonnes techniques ou cibles
    ignore = ['id', 'price', 'images', 'type_transaction', 'dataset_source', 'Unnamed: 0']
    
    num_cols = [c for c in df_sample.columns if df_sample[c].dtype != 'object' and c not in ignore]
    text_cols = [c for c in df_sample.columns if df_sample[c].dtype == 'object' and c not in ignore]
    return num_cols, text_cols


def prepare_scaler_from_subset(csv_folder, num_cols, sample_files=5, sep=";"):
    """Fit le scaler sur un échantillon de fichiers."""
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    if not csv_files:
        raise ValueError(f"Aucun CSV dans {csv_folder}")

    files_to_read = csv_files[:sample_files]
    data_values = []
    print(f"[SCALER] Fitting sur {len(files_to_read)} fichiers...")

    for f in files_to_read:
        df = pd.read_csv(f, sep=sep, usecols=num_cols)
        data_values.append(df.fillna(0).values)
     
    full_sample = np.vstack(data_values)
    scaler = StandardScaler()
    scaler.fit(full_sample)
    print("[SCALER] Prêt.")
    return scaler


# --- DATASET PYTORCH ---
class RealEstateDataset(Dataset):
    def __init__(self, csv_folder, img_dir, tokenizer, scaler, num_cols, text_cols, is_train=True, sep=";"):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.target_size = (224, 224)
        self.text_cols = text_cols

        # 1. Chargement de TOUS les CSV en mémoire
        files = glob.glob(os.path.join(csv_folder, "*.csv"))
        if not files:
            raise ValueError(f"Aucun CSV trouvé dans {csv_folder}")
            
        print(f"[DATASET] Chargement de {len(files)} fichiers CSV...")
        df_list = [pd.read_csv(f, sep=sep, dtype={'id': str}) for f in files]
        self.df = pd.concat(df_list, ignore_index=True)
        
        # 2. Préparation Tabulaire
        print("[DATASET] Normalisation des données tabulaires...")
        X_tab = self.df[num_cols].fillna(0.0).values
        self.tab_data = scaler.transform(X_tab).astype(np.float32)
        
        # 3. Pré-calcul des Cibles (Targets)
        print("[DATASET] Pré-calcul des cibles...")
        self.targets = torch.zeros((len(self.df), 2), dtype=torch.float32)
        self.masks = torch.zeros((len(self.df), 2), dtype=torch.float32)

        prices = pd.to_numeric(self.df['price'], errors='coerce').fillna(0).values
        sources = self.df['dataset_source'].fillna(0).astype(int).values
        
        # Boucle rapide
        for i in range(len(self.df)):
            val_log = math.log1p(max(0, prices[i]))
            src = sources[i]
            
            if src == 0: # ACHAT
                self.targets[i, 0] = val_log
                self.masks[i, 0] = 1.0
            elif src == 1: # LOCATION
                self.targets[i, 1] = val_log
                self.masks[i, 1] = 1.0
            # Si src est autre chose, mask reste à 0 (ignoré par la loss)
                
        print(f"[DATASET] Initialisé avec {len(self.df)} annonces.")


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_str = str(row['id'])

        # A. Images
        images = self._load_images(id_str)

        # B. Texte
        text_raw = " ".join([str(row[c]) for c in self.text_cols])
        tokens = self.tokenizer(
            text_raw, 
            padding='max_length', 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )

        return {
            'images': images,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'tab_data': torch.tensor(self.tab_data[idx], dtype=torch.float32),
            'target': self.targets[idx],
            'mask': self.masks[idx]
        }


    def _load_images(self, id_str):
        path = os.path.join(self.img_dir, id_str)
        tensors = []
        valid_ext = (".jpg", ".png", ".jpeg")

        if os.path.isdir(path):
            files = sorted([f for f in os.listdir(path) if f.lower().endswith(valid_ext)])

            for f in files:
                img = cv2.imread(os.path.join(path, f))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.target_size)
                    img = img.astype(np.float32) / 255.0
                    # Normalisation ImageNet
                    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                    img = np.transpose(img, (2, 0, 1)) 
                    tensors.append(torch.tensor(img, dtype=torch.float32))

        if not tensors:
            tensors.append(torch.zeros(3, *self.target_size))

        return torch.stack(tensors)


def real_estate_collate_fn(batch):
    """
    Assemble une liste d'éléments individuels en un batch.
    Gère le padding des images.
    """
    # 1. Padding Images
    max_imgs = max([item['images'].shape[0] for item in batch])
    img_dims = batch[0]['images'].shape[1:]

    padded_images = []
    for item in batch:
        imgs = item['images']
        diff = max_imgs - imgs.shape[0]
        if diff > 0:
            pad = torch.zeros((diff, *img_dims), dtype=torch.float32)
            imgs = torch.cat([imgs, pad], dim=0)
        padded_images.append(imgs)
        
    images_tensor = torch.stack(padded_images)

    # 2. Stack simple pour le reste
    return {
        'images': images_tensor,
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'tab_data': torch.stack([item['tab_data'] for item in batch]),
        'targets': torch.stack([item['target'] for item in batch]),
        'masks': torch.stack([item['mask'] for item in batch])
    }
