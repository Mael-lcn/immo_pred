import os
import argparse
import pandas as pd
import glob
import time
from tqdm import tqdm
import multiprocessing
from functools import partial
import csv



# Définition des colonnes OHE à retirer du vocabulaire final des FEATURES
EXCLUSIONS = {"bathtub", "second_bathroom", "sold_rented"}


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

        # 2. EXTRACTION ATOMIQUE
        # EXTÉRIEUR (Séparateur: ,)
        # Utilisation de str.split().explode().dropna() pour obtenir des éléments atomiques uniques
        ext_set = set(ext_series.str.split(',').explode().dropna().tolist())
        
        # FEATURES (Séparateur: |)
        feat_set = set(feat_series.str.split('|').explode().dropna().tolist())

        # 3. NETTOYAGE (Séparé)
        ext_set.discard("") 
        ext_set.discard(" ")
        feat_set.discard("")
        feat_set.discard(" ")

        # On retourne les deux sets distincts
        return (ext_set, feat_set) 

    except Exception as e:
        print(f"[WARN SCAN] Échec sur {file_path}: {e}")
        return (set(), set())


# =============================================================================
# WORKER 2 : APPLICATION DU OHE ET OVERWRITE (DEUX SETS)
# =============================================================================
def process_worker(file_path, global_ext_cols, global_feat_cols):
    """
    Charge un fichier, applique le OHE en utilisant les séparateurs natifs,
    aligne sur les deux listes de colonnes globales, et écrase le fichier.
    """
    try:
        # 1. Lecture complète 
        # Note: Pas besoin de forcer dtype=str si le script de filtrage a bien utilisé astype(str)
        df = pd.read_csv(file_path, sep=";")

        # 2. Préparation (remplacer les NaN)
        df['exterior_access'] = df['exterior_access'].fillna("")
        df['special_features'] = df['special_features'].fillna("")

        # 3. One-Hot Encoding LOCAL (avec les séparateurs natifs corrects)
        local_dummies_ext = df['exterior_access'].str.get_dummies(sep=',') # Sép. Virgule
        local_dummies_feat = df['special_features'].str.get_dummies(sep='|') # Sép. Pipe

        # 4. ALIGNEMENT GLOBAL : (Alignement sur DEUX listes globales différentes)
        # Chaque bloc est aligné sur sa propre liste de vocabulaire
        aligned_ext = local_dummies_ext.reindex(columns=global_ext_cols, fill_value=0)
        aligned_feat = local_dummies_feat.reindex(columns=global_feat_cols, fill_value=0)

        # 5. Renommage (Préfixes)
        aligned_ext = aligned_ext.add_prefix('ext_')
        aligned_feat = aligned_feat.add_prefix('feat_')

        # 6. Concaténation
        df_final = pd.concat([df.drop(columns=['exterior_access', 'special_features'], errors='ignore'), 
                              aligned_ext, 
                              aligned_feat], axis=1)

        # 7. Écriture (Overwrite)
        df_final.to_csv(file_path, sep=";", index=False, quoting=csv.QUOTE_MINIMAL)

        return 1
    except Exception as e:
        print(f"[ERROR ECRITURE] Écriture échouée pour {file_path}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="OHE In-Place avec alignement global des colonnes (2 sets).")
    parser.add_argument('-i', '--input', type=str, default='../output/csv', 
                        help="Dossier racine contenant les CSV (traitement récursif)")
    parser.add_argument('-w', '--workers', type=int, default=max(1, multiprocessing.cpu_count()-1))
    
    args = parser.parse_args()

    t0 = time.time()

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
        
        # Consolidation des deux sets séparément
        for (ext_set, feat_set) in results:
            global_ext_set.update(ext_set)
            global_feat_set.update(feat_set)

    # 2. FILTRAGE : Exclusion des features
    global_feat_set = global_feat_set.difference(EXCLUSIONS)

    # 3. Conversion et tri pour l'alignement
    global_ext_list = sorted(list(global_ext_set))
    global_feat_list = sorted(list(global_feat_set))

    # AFFICHAGE DU VOCABULAIRE FINAL
    print("\n--- VOCABULAIRE FINAL SÉPARÉ ---")
    
    print(f"Colonnes 'EXTERIOR' (Préfixe ext_) : {len(global_ext_list)} éléments")
    print(f"> Aperçu : {global_ext_list[:5]}...")
    
    print(f"Colonnes 'FEATURES' (Préfixe feat_) : {len(global_feat_list)} éléments")
    print(f"> Aperçu : {global_feat_list[:5]}...")
    print(f"[NOTE] {len(EXCLUSIONS)} features exclues : {list(EXCLUSIONS)}")

    # ---------------------------------------------------------
    # ÉTAPE 2 : ÉCRITURE
    # ---------------------------------------------------------
    print("\n--- Étape 2 : Standardisation et Écriture ---")

    # Préparation de la fonction worker avec les DEUX listes globales
    process_func = partial(process_worker, global_ext_cols=global_ext_list, global_feat_cols=global_feat_list)

    success_count = 0
    with multiprocessing.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_func, files), total=len(files), desc="Écriture"))
        success_count = sum(results)

    dt = time.time() - t0
    print(f"\nTerminé en {dt:.2f}s.")
    print(f"Fichiers traités avec succès : {success_count}/{len(files)}")


if __name__ == '__main__':
    main()
