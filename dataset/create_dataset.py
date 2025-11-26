import os
import argparse
import pandas as pd
import time
import glob
from tqdm import tqdm
from functools import partial
import multiprocessing
import csv


rename_dict = {
    "id": "id",
    "titre": "titre",
    "type_bien": "property_type", 
    "etat_bien": "property_status",
    "prix": "price",
    "ville": "city",
    "region": "region",
    "codePostal": "postal_code",
    "departement": "department",
    "nb_pieces": "num_rooms", #entre 1 et 100 
    "nb_chambres": "num_bedrooms", #entre 1 et 80
    "nb_salleDeBains": "num_bathrooms", #entre 0 et 50
    "classe_energetique": "energy_rating", # A-> F
    "orientation": "orientation", #points cardinaux 
    "nb_placesParking": "num_parking_spaces", # entre 0 et 50
    "surface_habitable": "living_area_sqm", #entre 10 et 850
    "surface_tolale_terrain": "total_land_area_sqm", #entre 0 et 8e5
    "nb_etages_Immeuble": "building_num_floors",  #entre 0 et 33
    "prix_metre_carre": "price_per_sqm", # entre 200 et 50000
    "annee_construction": "year_built", # entre 1700 et 2025
    "specificites": "features", 
    "images_urls": "images", #+ que le nombres de pieces 
    "description": "description",
}

colonnes_cibles = list(rename_dict.values())



# Un peu éclaté
def filtre(df):
    """
    Nettoie et transforme la DataFrame :
      - Remplit total_land_area_sqm par 0 si présent
      - Supprime les lignes contenant des valeurs NaN (après conversions nécessaires)
      - Supprime les lignes où nb d'images < nb de pièces (images compte les '|' ; si champ vide => 0)
      - Supprime les doublons
      - Supprime les colonnes vides (toutes valeurs NaN ou '')
      - Réinitialise l'index

    Retourne : (df_nettoye, stats_dict)
    """
    stats = {
        "initial_rows": len(df) if df is not None else 0,
        "final_rows": 0,
        "na_removed": 0,
        "dup_removed": 0,
        "empty_removed": 0,
        "images_removed": 0
    }

    if df is None or df.empty:
        return df, stats

    # --- Drop rows with any NaN (après conversions utiles)
    before_dropna = len(df)
    df = df.dropna()
    stats["na_removed"] = before_dropna - len(df)

    # --- Supprimer lignes où le nombre d'images < num_rooms

    # images: compter '|' uniquement si la chaine non vide, sinon 0
    images_str = df['images'].fillna('').astype(str)
    num_images = images_str.str.count(r'\|') + (images_str != '').astype(int)

    # convertir num_rooms en int (valeurs non numériques deviennent NaN)
    num_rooms_numeric = pd.to_numeric(df['num_rooms'])

    # Pour la comparaison, si num_rooms est NaN on considère la ligne invalide -> on la supprimera (mask False)
    mask_valid = num_rooms_numeric.notna()  # on garde seulement si num_rooms convertible
    mask_images = (num_images >= num_rooms_numeric.fillna(-1))  # comparaison sécurisée

    # Final mask : both valid num_rooms AND images >= num_rooms
    final_mask = mask_valid & mask_images

    images_removed_count = (~final_mask).sum()
    stats["images_removed"] = int(images_removed_count)
    df = df[final_mask]

    # --- Doublons
    before_dup = len(df)
    df = df.drop_duplicates(ignore_index=True)
    stats["dup_removed"] = before_dup - len(df)

    # --- Colonnes vides (toutes NaN ou toutes chaîne vide '')
    empty_cols = []
    for col in df.columns:
        # si toute la colonne est NaN
        if df[col].isna().all():
            empty_cols.append(col)
            continue
        # ou toute la colonne est '', après cast en str (attention aux NaN déjà éliminés)
        # utiliser astype(str) n'est pas idéal si valeurs NaN existent; on a déjà dropna
        if (df[col].astype(str) == '').all():
            empty_cols.append(col)

    if empty_cols:
        df = df.drop(columns=empty_cols)
    stats["empty_removed"] = len(empty_cols)

    # --- Final
    df = df.reset_index(drop=True)
    stats["final_rows"] = len(df)

    return df, stats


def worker(csv_path, output_dir):
    """
    Traite un fichier CSV :
      - lecture, renommage colonnes
      - fillna(0) pour total_land_area_sqm et num_parking_spaces si présent
      - filtrage via filtre()
      - écrit CSV filtré et retourne stats (dict)
    """
    default_stats = {
        "initial_rows": 0,
        "final_rows": 0,
        "na_removed": 0,
        "dup_removed": 0,
        "empty_removed": 0,
        "images_removed": 0
    }

    try:
        df = pd.read_csv(csv_path, sep=";", quotechar='"', dtype=str)
        if df.empty:
            return default_stats

        stats = {"initial_rows": len(df)}
        # rename columns
        df.rename(columns=rename_dict, inplace=True)

        # garder seulement colonnes cibles présentes
        final_cols_list = [c for c in colonnes_cibles]
        df = df[final_cols_list]

        # fillna seulement si colonnes présentes
        df['total_land_area_sqm'] = pd.to_numeric(df['total_land_area_sqm'], errors='coerce').fillna(0)
        df['num_parking_spaces'] = pd.to_numeric(df['num_parking_spaces'], errors='coerce').fillna(0)

        # Appliquer le filtre
        df_filtered, filtre_stats = filtre(df)

        # Mettre à jour stats initiales
        stats.update(filtre_stats)

        # Écrire le CSV filtré
        try:
            output_file = os.path.join(output_dir, os.path.basename(csv_path))
            df_filtered.to_csv(output_file, sep=";", index=False, quoting=csv.QUOTE_MINIMAL)
        except Exception as e:
            error_file = os.path.join(output_dir, "erreur.txt")
            with open(error_file, "a", encoding="utf-8") as f:
                f.write(f"[ERROR WRITE] {csv_path} : {e}\n")

    except Exception as e:
        error_file = os.path.join(output_dir, "erreur.txt")
        with open(error_file, "a", encoding="utf-8") as f:
            f.write(f"[ERROR READ] {csv_path} : {e}\n")
        stats = default_stats

    return stats



def run(args):
    t0 = time.monotonic()
    os.makedirs(args.output, exist_ok=True)

    csv_list = glob.glob(os.path.join(args.input, '*.csv'))

    if not csv_list:
        print(f"[WARN] Aucun fichier CSV trouvé dans : {args.input}")
        return

    num_processes = min(args.workers, len(csv_list))
    print(f"Traitement {len(csv_list)} fichiers -> {len(csv_list)} lots, processes={num_processes}")

    fun = partial(worker, output_dir=args.output)

    stats_global = {
        "initial_rows": 0,
        "final_rows": 0,
        "na_removed": 0,
        "dup_removed": 0,
        "empty_removed": 0,
        "images_removed": 0
    }

    # Pool multiprocessing
    with multiprocessing.Pool(processes=num_processes) as pool:
        for batch_stats in tqdm(pool.imap_unordered(fun, csv_list), total=len(csv_list)):
            if not batch_stats:
                continue
            # batch_stats est un dict
            for key in stats_global:
                stats_global[key] += int(batch_stats.get(key, 0))

    dt = time.monotonic() - t0
    print(f"\nTemps total: {dt:.2f}s. CSV filtrés écrits dans : {args.output}")

    erreur_file = os.path.join(args.output, "erreur.txt")
    if os.path.exists(erreur_file):
        print(f"[WARN] Certaines erreurs ont été enregistrées dans : {erreur_file}")

    # Résumé final
    print("\n=== Résumé global du nettoyage ===")
    print(f"Lignes initiales: {stats_global['initial_rows']}")
    print(f"Lignes finales  : {stats_global['final_rows']}")
    print(f"Lignes supprimées (NaN)   : {stats_global['na_removed']}")
    print(f"Lignes supprimées (doublons) : {stats_global['dup_removed']}")
    print(f"Colonnes vides supprimées     : {stats_global['empty_removed']}")
    print(f"Lignes avec moins d'images que de pièces  : {stats_global['images_removed']}")
    kept_pct = (stats_global['final_rows'] / stats_global['initial_rows'] * 100)
    print(f"Pourcentage de données gardée : {kept_pct:.2f}%")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='../data/Lbc/achat/')
    parser.add_argument('-o', '--output', type=str, default='output/csv')
    parser.add_argument('-w', '--workers', type=int, default=max(1, multiprocessing.cpu_count()-1))
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
