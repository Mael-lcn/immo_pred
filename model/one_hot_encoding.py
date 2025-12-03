import os
import argparse
import pandas as pd
import glob
import time
from tqdm import tqdm
import multiprocessing
from functools import partial
import csv

from sklearn.preprocessing import StandardScaler



# Définition des colonnes OHE à retirer du vocabulaire final des FEATURES
EXCLUSIONS = {"bathtub", "second_bathroom", "sold_rented"}

# Mappings Ordinaux
DPE_MAP = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
ETAT_MAP = {
    'Rénové': 4, 'Très bon état': 3, 'Bon état': 2,
    'À rafraichir': 1, 'Travaux à prévoir': 1,
}
PROPERTY_MAP = {'Appartement': 1, 'Maison': 0}


# =============================================================================
# WORKER 1 : SCAN DU VOCABULAIRE (ATOMICITÉ PAR EXPLODE)
# =============================================================================
def scan_worker(file_path):
    """
    Lit un fichier et extrait les sets atomiques en utilisant split/explode.
    Retourne deux sets distincts : (ext_set, feat_set).
    """
    try:
        # Lecture
        df = pd.read_csv(file_path, sep=";", usecols=['exterior_access', 'special_features'])

        # Préparation/Conversion (Sécurité)
        ext_series = df['exterior_access'].fillna("").astype(str)
        feat_series = df['special_features'].fillna("").astype(str)

        # EXTRACTION ATOMIQUE
        ext_set = set(ext_series.str.split(',').explode().str.strip().dropna().tolist())
        feat_set = set(feat_series.str.split('|').explode().str.strip().dropna().tolist())

        # NETTOYAGE
        ext_set.discard("") 
        feat_set.discard("")

        return (ext_set, feat_set) 

    except Exception as e:
        print(f"[WARN SCAN] Échec sur {file_path}: {e}")
        return (set(), set())


# =============================================================================
# WORKER 2 : APPLICATION DU OHE ET SAUVEGARDE (NOUVEAU FICHIER)
# =============================================================================
def process_worker(file_path, global_ext_cols, global_feat_cols, input_base_dir, output_base_dir):
    """
    Charge un fichier, applique le OHE, aligne sur les colonnes globales,
    et écrit dans le dossier de sortie en reproduisant l'arborescence.
    """
    try:
        # 1. Lecture complète 
        df = pd.read_csv(file_path, sep=";")

        # 2. Préparation (remplacer les NaN)
        df['exterior_access'] = df['exterior_access'].fillna("")
        df['special_features'] = df['special_features'].fillna("")

        df['energy_rating'] = df['energy_rating'].map(DPE_MAP)
        df['property_status'] = df['property_status'].map(ETAT_MAP)
        df['property_type'] = df['property_type'].map(PROPERTY_MAP)

        # 3. One-Hot Encoding LOCAL
        local_dummies_ext = df['exterior_access'].str.get_dummies(sep=',') # Sép. Virgule
        local_dummies_feat = df['special_features'].str.get_dummies(sep='|') # Sép. Pipe

        # 4. ALIGNEMENT GLOBAL
        aligned_ext = local_dummies_ext.reindex(columns=global_ext_cols, fill_value=0)
        aligned_feat = local_dummies_feat.reindex(columns=global_feat_cols, fill_value=0)

        # 5. Renommage (Préfixes)
        aligned_ext = aligned_ext.add_prefix('ext_')
        aligned_feat = aligned_feat.add_prefix('feat_')

        # 6. Concaténation
        df_final = pd.concat([df.drop(columns=['exterior_access', 'special_features'], errors='ignore'), 
                              aligned_ext, 
                              aligned_feat], axis=1)

        # 7. GESTION DES CHEMINS (Input -> Output)
        # Calcul du chemin relatif (ex: 'sous_dossier/fichier.csv') par rapport au dossier input racine
        rel_path = os.path.relpath(file_path, input_base_dir)
        # Création du chemin final complet
        target_path = os.path.join(output_base_dir, rel_path)
        
        # Création du dossier parent s'il n'existe pas
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # 8. Écriture
        df_final.to_csv(target_path, sep=";", index=False, quoting=csv.QUOTE_MINIMAL)

        return 1
    except Exception as e:
        print(f"[ERROR ECRITURE] Écriture échouée pour {file_path}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="OHE avec alignement global et écriture dans dossier sortie.")
    parser.add_argument('-i', '--input', type=str, default='../output/raw_csv')
    parser.add_argument('-o', '--output', type=str, default='../output/processed_csv')
    parser.add_argument('-w', '--workers', type=int, default=max(1, multiprocessing.cpu_count()-1))

    args = parser.parse_args()

    t0 = time.time()
    
    # Vérification dossier input
    if not os.path.exists(args.input):
        print(f"Erreur: Le dossier d'entrée {args.input} n'existe pas.")
        return

    # Création dossier output racine si besoin
    if not os.path.exists(args.output):
        print(f"Création du dossier de sortie : {args.output}")
        os.makedirs(args.output, exist_ok=True)

    files = glob.glob(os.path.join(args.input, '**', '*.csv'), recursive=True)
    if not files:
        print("Aucun fichier trouvé.")
        return

    print(f"Démarrage du traitement de {len(files)} fichiers avec {args.workers} workers.")

    # ---------------------------------------------------------
    # ÉTAPE 1 : SCAN GLOBAL (CONSOLIDER DEUX SETS SÉPARÉS)
    # ---------------------------------------------------------
    print("\n--- Étape 1 : Analyse du vocabulaire global ---")

    global_ext_set = set()
    global_feat_set = set()

    with multiprocessing.Pool(args.workers) as pool:
        # Les workers retournent (ext_set, feat_set) pour chaque fichier
        results = list(tqdm(pool.imap_unordered(scan_worker, files), total=len(files), desc="Scan"))

        # Consolidation
        for (ext_set, feat_set) in results:
            global_ext_set.update(ext_set)
            global_feat_set.update(feat_set)

    # FILTRAGE
    global_feat_set = global_feat_set.difference(EXCLUSIONS)

    # Conversion et tri
    global_ext_list = sorted(list(global_ext_set))
    global_feat_list = sorted(list(global_feat_set))

    print(f"Colonnes 'EXTERIOR' : {len(global_ext_list)} éléments")
    print(f"> : {list(global_ext_list)} éléments\n")
    print(f"Colonnes 'FEATURES' : {len(global_feat_list)} éléments")
    print(f">' : {list(global_feat_list)} éléments")

    # ---------------------------------------------------------
    # ÉTAPE 2 : ÉCRITURE
    # ---------------------------------------------------------
    print(f"\n--- Étape 2 : Standardisation et Écriture vers {args.output} ---")

    # Préparation de la fonction worker avec les chemins racine
    process_func = partial(
        process_worker, 
        global_ext_cols=global_ext_list, 
        global_feat_cols=global_feat_list, 
        input_base_dir=args.input,
        output_base_dir=args.output
    )

    success_count = 0
    with multiprocessing.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_func, files), total=len(files), desc="Écriture"))
        success_count = sum(results)

    dt = time.time() - t0
    print(f"\nTerminé en {dt:.2f}s.")
    print(f"Fichiers traités avec succès : {success_count}/{len(files)}")


if __name__ == '__main__':
    main()
