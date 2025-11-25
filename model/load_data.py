"""
Module de gestion des données "Streaming" (Low Memory).
Charge les CSV un par un et génère des batchs à la volée.

Flux de données :
Disque (Multi CSVs) -> RAM (1 seul CSV) -> Batch (8 lignes) -> GPU
"""

import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler



# ----------------- 1. FONCTIONS UTILITAIRES -----------------

def load_images_for_batch(image_dir, list_ids, target_size=(224, 224)):
    """
    Charge TOUTES les images disponibles pour chaque ID du batch,
    avec une limite de sécurité pour la VRAM.
    
    Stratégie : Padding Dynamique.
    Le tenseur final aura la taille : (Batch_Size, Max_Images_In_This_Batch, 3, H, W)
    """
    valid_ext = (".jpg", ".png", ".jpeg")
    
    # 1. On charge d'abord tout en mémoire (Listes de Tensors)
    batch_data = [] # Liste de listes de tensors
    global_max_len = 0 # Le record d'images dans CE batch
    
    for id_str in list_ids:
        path_annonce = os.path.join(image_dir, str(id_str))
        img_tensor_list = []
        
        if os.path.isdir(path_annonce):
            # On trie pour avoir toujours le même ordre
            files = sorted([f for f in os.listdir(path_annonce) if f.lower().endswith(valid_ext)])

            for fname in files:
                img_path = os.path.join(path_annonce, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, target_size)
                    img = img.astype(np.float32) / 255.0
                    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                    img = np.transpose(img, (2, 0, 1))
                    img_tensor_list.append(torch.tensor(img, dtype=torch.float32))

        # Si aucune image valide, on met une image noire (placeholder)
        if len(img_tensor_list) == 0:
            img_tensor_list.append(torch.zeros(3, *target_size))

        # Mise à jour du record pour ce batch
        global_max_len = max(global_max_len, len(img_tensor_list))
        batch_data.append(img_tensor_list)

    # 2. On applique le Padding pour uniformiser ce batch
    final_batch = []

    for img_list in batch_data:
        current_len = len(img_list)
        diff = global_max_len - current_len

        # Si on a moins d'images que le max du batch, on ajoute des zéros
        if diff > 0:
            # On crée N images noires d'un coup
            padding = [torch.zeros(3, *target_size) for _ in range(diff)]
            img_list.extend(padding)
            
        # On empile les images de cette annonce (N, 3, H, W)
        final_batch.append(torch.stack(img_list))

    # 3. On empile les annonces entre elles (Batch, N, 3, H, W)
    return torch.stack(final_batch)


def get_cols_config(csv_path, sep=";"):
    """
    Lit juste la première ligne d'un CSV pour détecter les colonnes.
    Très léger en mémoire.
    """
    df_sample = pd.read_csv(csv_path, sep=sep, nrows=5)
    ignore = ['id', 'price', 'images', 'type_transaction']
    num_cols = [c for c in df_sample.columns if df_sample[c].dtype != 'object' and c not in ignore]
    text_cols = [c for c in df_sample.columns if df_sample[c].dtype == 'object' and c not in ignore]
    return num_cols, text_cols


def prepare_scaler_from_subset(csv_folder, num_cols, sample_files=3, sep=";"):
    """
    Entraîne le Scaler sur un sous-ensemble des données (ex: les 3 premiers fichiers CSV).
    On ne peut pas fit sur tout le dataset s'il ne rentre pas en RAM, donc on approxime.
    """
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    if not csv_files:
        raise ValueError(f"Aucun CSV trouvé dans {csv_folder}")

    # On prend quelques fichiers au hasard pour avoir une bonne distribution
    files_to_read = csv_files[:sample_files] 
    data_values = []

    print(f"[SCALER] Fitting sur {len(files_to_read)} fichiers...")
    for f in files_to_read:
        df = pd.read_csv(f, sep=sep, usecols=num_cols)
        data_values.append(df.fillna(0).values)

    # Concaténation de l'échantillon
    full_sample = np.vstack(data_values)

    scaler = StandardScaler()
    scaler.fit(full_sample)
    print(f"[SCALER] Moyennes calculées. Prêt.")
    return scaler


# ----------------- 2. LE GÉNÉRATEUR -----------------
def create_streaming_generator(csv_folder, image_dir, tokenizer, batch_size=8, scaler=None, is_train=True, sep=";"):
    """
    Générateur principal.
    1. Liste tous les CSV du dossier.
    2. Charge un CSV entier en RAM.
    3. Découpe ce CSV en batchs.
    4. Charge les images du batch.
    5. Yield le batch.
    6. Passe au CSV suivant (et libère la RAM du précédent).
    """

    # 1. Lister les fichiers
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    if not csv_files:
        raise ValueError("Dossier CSV vide ou chemin incorrect.")
        
    # Mélange l'ordre des fichiers CSV (Randomisation Macro)
    if is_train:
        np.random.shuffle(csv_files)

    # 2. Détection config (basée sur le premier fichier trouvé)
    num_cols, text_cols = get_cols_config(csv_files[0], sep=sep)

    # Gestion Scaler automatique si non fourni (Fit sur le premier fichier)
    if is_train and scaler is None:
        scaler = prepare_scaler_from_subset(csv_folder, num_cols, sample_files=2)

    # --- BOUCLE SUR LES FICHIERS (Macro-Batch) ---
    for csv_file in csv_files:
        # print(f"Processing file: {os.path.basename(csv_file)}") # Debug optionnel
        
        try:
            # On charge UN fichier complet en RAM
            # (Supposons que chaque CSV fait < 1Go, c'est ok pour la plupart des RAMs)
            df_current = pd.read_csv(csv_file, sep=sep)
        except Exception as e:
            print(f"[WARN] Erreur lecture {csv_file}: {e}")
            continue

        # Mélange des lignes DANS le fichier (Randomisation Micro)
        if is_train:
            df_current = df_current.sample(frac=1).reset_index(drop=True)

        # Conversion en dict pour vitesse
        records = df_current.to_dict('records')
        num_samples = len(records)
    
        # --- BOUCLE SUR LES LIGNES (Micro-Batch) ---
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_raw = records[start_idx:end_idx]
            
            # --- A. Chargement Images (LAZINESS: ON LE FAIT ICI) ---
            batch_ids = [r['id'] for r in batch_raw]
            images_tensor = load_images_for_batch(image_dir, batch_ids)
            
            # --- B. Texte ---
            batch_texts = [" ".join([str(r[c]) for c in text_cols]) for r in batch_raw]
            txt_out = tokenizer(batch_texts, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
            
            # --- C. Tabulaire ---
            batch_nums = []
            for r in batch_raw:
                vals = [float(r[c]) if not pd.isna(r[c]) else 0.0 for c in num_cols]
                batch_nums.append(vals)
            
            batch_nums = np.array(batch_nums)
            batch_nums_norm = scaler.transform(batch_nums)
            tab_tensor = torch.tensor(batch_nums_norm, dtype=torch.float32)
            
            # --- D. Cibles ---
            targets = torch.zeros((len(batch_raw), 2), dtype=torch.float32)
            masks = torch.zeros((len(batch_raw), 2), dtype=torch.float32)

            for i, r in enumerate(batch_raw):
                try:
                    price = float(r['price'])
                    val_log = math.log1p(max(0, price))
                    t_type = str(r['type_transaction']).lower()
                    
                    if 'vente' in t_type or 'buy' in t_type:
                        targets[i, 0] = val_log
                        masks[i, 0] = 1.0
                    elif 'loc' in t_type or 'rent' in t_type:
                        targets[i, 1] = val_log
                        masks[i, 1] = 1.0
                except:
                    pass # Sécurité si données corrompues

            yield {
                'images': images_tensor,
                'input_ids': txt_out['input_ids'],
                'attention_mask': txt_out['attention_mask'],
                'tab_data': tab_tensor,
                'targets': targets,
                'masks': masks
            }
        
        # Suppression explicite pour aider le Garbage Collector
        del df_current
        del records
