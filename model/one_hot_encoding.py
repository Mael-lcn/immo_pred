import os
import argparse
import pandas as pd
import glob
import time
from tqdm import tqdm
import multiprocessing
from functools import partial
import csv
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

# Colonnes à exclure du vocabulaire final (bruit ou redondance)
EXCLUSIONS = {"bathtub", "second_bathroom", "sold_rented"}

# Mappings Ordinaux (Les NaN resteront NaN après mapping, c'est voulu)
DPE_MAP = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
ETAT_MAP = {
    'Rénové': 4, 'Très bon état': 3, 'Bon état': 2,
    'À rafraichir': 1, 'Travaux à prévoir': 0,
}
PROPERTY_MAP = {'Appartement': 1, 'Maison': 0}

# Colonnes acceptant les NaN -> On génère un flag "_is_missing"
NAN_TOLERATED_COLS = [
    "energy_rating",
    "year_built",
    "num_bedrooms",
    "num_bathrooms",
    "property_status",
    "orientation"
]

# =============================================================================
# WORKER 1 : SCAN DU VOCABULAIRE (MODE CONFIANT)
# =============================================================================
def scan_worker(file_path):
    try:
        # Lecture optimisée : on ne charge que le nécessaire pour le scan
        df = pd.read_csv(file_path, sep=";", usecols=['exterior_access', 'special_features', 'region'])

        # Pas de check, on assume que les colonnes sont là
        ext_series = df['exterior_access'].fillna("").astype(str)
        feat_series = df['special_features'].fillna("").astype(str)
        region_series = df['region'].dropna().astype(str)

        # Extraction atomique
        ext_set = set(ext_series.str.split(',').explode().str.strip().dropna().tolist())
        feat_set = set(feat_series.str.split('|').explode().str.strip().dropna().tolist())
        region_set = set(region_series.tolist())

        # Nettoyage
        ext_set.discard("")
        feat_set.discard("")
        
        return (ext_set, feat_set, region_set)

    except Exception as e:
        print(f"[FATAL SCAN] {file_path}: {e}")
        return (set(), set(), set())


# =============================================================================
# WORKER 2 : PROCESSING COMPLET (MODE CONFIANT)
# =============================================================================
def process_worker(file_path, global_ext_cols, global_feat_cols, global_region_list, input_base_dir, output_base_dir):
    try:
        # 1. Lecture
        df = pd.read_csv(file_path, sep=";")

        # ---------------------------------------------------------
        # A. GESTION DES FLAGS (IS_MISSING)
        # ---------------------------------------------------------
        # On assume que toutes ces colonnes existent.
        # On crée le flag AVANT toute modification.
        for col in NAN_TOLERATED_COLS:
            # Création du flag binaire (1 si NaN, 0 sinon)
            df[f'{col}_is_missing'] = df[col].isna().astype(int)
            
            # NOTE IMPORTANTE : On ne remplit PAS la colonne originale ici.
            # On la laisse à NaN pour que le DataLoader puisse calculer
            # la médiane/mode sur le Train Set uniquement plus tard.

        # ---------------------------------------------------------
        # B. MAPPINGS ORDINAUX & BINAIRES
        # ---------------------------------------------------------
        df['energy_rating'] = df['energy_rating'].map(DPE_MAP)
        df['property_status'] = df['property_status'].map(ETAT_MAP)
        df['property_type'] = df['property_type'].map(PROPERTY_MAP)

        # ---------------------------------------------------------
        # C. PRÉPARATION TEXTE POUR OHE
        # ---------------------------------------------------------
        # On remplit les NaN par vide juste pour que le split fonctionne
        df['exterior_access'] = df['exterior_access'].fillna("")
        df['special_features'] = df['special_features'].fillna("")
        
        # Pour la région, on force le type Categorical aligné sur le global direct
        df['region'] = pd.Categorical(df['region'], categories=global_region_list)

        # ---------------------------------------------------------
        # D. ONE-HOT ENCODING & ALIGNEMENT
        # ---------------------------------------------------------
        # Multi-Label (Séparateurs , et |)
        local_dummies_ext = df['exterior_access'].str.get_dummies(sep=',')
        local_dummies_feat = df['special_features'].str.get_dummies(sep='|')
        
        # Mono-Label (Region)
        local_dummies_region = pd.get_dummies(df['region'], prefix="region")

        # Alignement strict sur le vocabulaire global
        # (reindex va créer des colonnes de 0 là où ça manque)
        aligned_ext = local_dummies_ext.reindex(columns=global_ext_cols, fill_value=0)
        aligned_feat = local_dummies_feat.reindex(columns=global_feat_cols, fill_value=0)
        
        # Pour les régions, on reconstruit les noms de colonnes attendus
        expected_region_cols = [f"region_{r}" for r in global_region_list]
        aligned_region = local_dummies_region.reindex(columns=expected_region_cols, fill_value=0)

        # Renommage pour éviter les collisions
        aligned_ext = aligned_ext.add_prefix('ext_')
        aligned_feat = aligned_feat.add_prefix('feat_')

        # ---------------------------------------------------------
        # E. ASSEMBLAGE & SAUVEGARDE
        # ---------------------------------------------------------
        cols_to_drop = ['exterior_access', 'special_features', 'region']
        
        df_final = pd.concat([
            df.drop(columns=cols_to_drop),
            aligned_ext,
            aligned_feat,
            aligned_region
        ], axis=1)

        # Calcul chemin de sortie
        rel_path = os.path.relpath(file_path, input_base_dir)
        target_path = os.path.join(output_base_dir, rel_path)
        
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        df_final.to_csv(target_path, sep=";", index=False, quoting=csv.QUOTE_MINIMAL)

        return 1

    except Exception as e:
        print(f"[ERROR PROCESSING] {file_path}: {e}")
        return 0


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="OHE Strict & Flags Missing")
    parser.add_argument('-i', '--input', type=str, default='../output/raw_csv')
    parser.add_argument('-o', '--output', type=str, default='../output/processed_csv')
    parser.add_argument('-w', '--workers', type=int, default=max(1, multiprocessing.cpu_count()-1))
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input introuvable: {args.input}")
        return

    # Recherche récursive
    files = glob.glob(os.path.join(args.input, '**', '*.csv'), recursive=True)
    if not files:
        print("Aucun CSV trouvé.")
        return

    print(f"Traitement de {len(files)} fichiers...")

    # 1. SCAN GLOBAL
    global_ext, global_feat, global_region = set(), set(), set()
    
    with multiprocessing.Pool(args.workers) as pool:
        for ext_s, feat_s, reg_s in tqdm(pool.imap_unordered(scan_worker, files), total=len(files), desc="1/2 Scan Vocabulaire"):
            global_ext.update(ext_s)
            global_feat.update(feat_s)
            global_region.update(reg_s)

    # Nettoyage Vocabulaire
    global_feat = global_feat.difference(EXCLUSIONS)
    global_region.discard(np.nan)
    global_region.discard("nan")
    global_region.discard("")

    # Tri pour garantir l'ordre des colonnes
    ext_list = sorted(list(global_ext))
    feat_list = sorted(list(global_feat))
    region_list = sorted(list(global_region))

    print(f"\nVocabulaire Global :")
    print(f"- Exterior Elements : {len(ext_list)}")
    print(f"- Special Features  : {len(feat_list)}")
    print(f"- Regions           : {len(region_list)}")

    # 2. PROCESSING DISTRIBUÉ
    process_func = partial(
        process_worker,
        global_ext_cols=ext_list,
        global_feat_cols=feat_list,
        global_region_list=region_list,
        input_base_dir=args.input,
        output_base_dir=args.output
    )

    with multiprocessing.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_func, files), total=len(files), desc="2/2 Encoding & Flags"))
    
    print(f"\nTerminé. Succès : {sum(results)} / {len(files)}")

if __name__ == '__main__':
    main()
