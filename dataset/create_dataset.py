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
    "nb_pieces": "num_rooms", 
    "nb_chambres": "num_bedrooms", 
    "nb_salleDeBains": "num_bathrooms", 
    "classe_energetique": "energy_rating", 
    "orientation": "orientation", 
    "nb_placesParking": "num_parking_spaces", 
    "surface_habitable": "living_area_sqm", 
    "surface_tolale_terrain": "total_land_area_sqm", 
    "nb_etages_Immeuble": "building_num_floors", 
    "prix_metre_carre": "price_per_sqm", 
    "annee_construction": "year_built", 
    "specificites": "features", 
    "images_urls": "images", 
    "description": "description",
}

colonnes_cibles = list(rename_dict.values())


def filtre(df):
    """
    Nettoie le DataFrame :
      - Filtre les plages numériques (supprime ce qui n'est pas un nombre standard).
      - Convertit en INT.
      - Filtre la classe énergétique et les images.
      - PAS de conversion de texte ("deux" -> 2).
    """
    if df is None or df.empty:
        return df

    # 1.Gestion des valeurs par défaut
    df["total_land_area_sqm"] = df["total_land_area_sqm"].fillna(0)
    df["num_parking_spaces"] = df["num_parking_spaces"].fillna(0)

    # 2. Drop les lignes contenants NaN
    df = df.dropna()

    # 3. Filtrage numérique
    numeric_rules = {
        "num_rooms": (1, 100),
        "num_bedrooms": (1, 80),
        "num_bathrooms": (0, 50),
        "num_parking_spaces": (0, 50),
        "living_area_sqm": (10, 850),
        "total_land_area_sqm": (0, 800000),
        "building_num_floors": (0, 33),
        "price_per_sqm": (200, 50000),
        "year_built": (1700, 2025)
    }

    for col, (mini, maxi) in numeric_rules.items():
        if col in df.columns:
            s_numeric = pd.to_numeric(df[col], errors='coerce')

            # Masque : on garde seulement les nombres valides dans la plage
            mask = s_numeric.between(mini, maxi)
            df = df[mask]

            # Conversion finale en INT
            df[col] = df[col].astype(int)

    # 4. Classe énergétique
    valid_ratings = ['A', 'B', 'C', 'D', 'E', 'F']
    df["energy_rating"] = df["energy_rating"].astype(str).str.upper().str.strip()
    df = df[df["energy_rating"].isin(valid_ratings)]

    # 5. Images vs pièces
    images_str = df['images'].astype(str)
    num_images = images_str.str.count(r'\|') + (images_str != '').astype(int)
    df = df[num_images >= df['num_rooms']]

    # 6. Finalisation
    df = df.drop_duplicates(ignore_index=True)
    df = df.reset_index(drop=True)

    return df



def worker(csv_path, output_dir):
    """
    Traite un fichier CSV et retourne le nombre de lignes avant/après pour les stats.
    """
    try:
        # Lecture
        df = pd.read_csv(csv_path, sep=";", quotechar='"', dtype=str)
        
        if df.empty:
            return 0, 0

        # Renommage
        df.rename(columns=rename_dict, inplace=True)

        # Sélection des colonnes cibles uniquement
        cols_presentes = [c for c in colonnes_cibles if c in df.columns]
        df = df[cols_presentes]

        nb_avant = len(df)

        # Application du filtre
        df_filtered = filtre(df)

        nb_apres = len(df_filtered)

        # Écriture
        output_file = os.path.join(output_dir, os.path.basename(csv_path))
        df_filtered.to_csv(output_file, sep=";", index=False, quoting=csv.QUOTE_MINIMAL)

        return nb_avant, nb_apres

    except Exception as e:
        print(f"[ERROR WORKER] {csv_path} : {e}\n")

        return 0, 0


def run(args):
    t0 = time.monotonic()
    os.makedirs(args.output, exist_ok=True)

    csv_list = glob.glob(os.path.join(args.input, '*.csv'))

    if not csv_list:
        print(f"[WARN] Aucun fichier CSV trouvé dans : {args.input}")
        return

    num_processes = min(args.workers, len(csv_list))
    print(f"Traitement de {len(csv_list)} fichiers avec {num_processes} processus...")

    fun = partial(worker, output_dir=args.output)

    # Variables globales pour le comptage
    total_raw_rows = 0
    total_clean_rows = 0

    # Exécution parallèle
    with multiprocessing.Pool(processes=num_processes) as pool:
        # On récupère les résultats (avant, apres) au fur et à mesure
        for result in tqdm(pool.imap_unordered(fun, csv_list), total=len(csv_list)):
            if result:
                avant, apres = result
                total_raw_rows += avant
                total_clean_rows += apres

    dt = time.monotonic() - t0
    
    # --- AFFICHAGE DU BILAN ---
    print("\n" + "="*40)
    print("BILAN DU NETTOYAGE")
    print("="*40)
    print(f"Fichiers traités     : {len(csv_list)}")
    print(f"Lignes AVANT filtres : {total_raw_rows:,}".replace(",", " "))
    print(f"Lignes APRES filtres : {total_clean_rows:,}".replace(",", " "))
    
    diff = total_raw_rows - total_clean_rows
    percent = (diff / total_raw_rows * 100) if total_raw_rows > 0 else 0
    
    print(f"Lignes supprimées    : {diff:,} (-{percent:.2f}%)".replace(",", " "))
    print("-" * 40)
    print(f"Temps total          : {dt:.2f}s")
    print(f"Dossier de sortie    : {args.output}")
    print("="*40)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='../../data/Lbc/achat/')
    parser.add_argument('-o', '--output', type=str, default='../output/csv')
    parser.add_argument('-w', '--workers', type=int, default=max(1, multiprocessing.cpu_count()-1))
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
