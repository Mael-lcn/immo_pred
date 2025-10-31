import os
import argparse
import pandas as pd
import time
import glob
from tqdm import tqdm
from functools import partial
import multiprocessing
import math



rename_dict_lbc = {
    "id": "id",
    "titre": "titre",
    "type_bien": "property_type",
    "etat_bien": "property_status",
    "nom_vendeur": "seller_name",
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
    "surface_totale_terrain": "total_land_area_sqm",
    "nb_etages_Immeuble": "building_num_floors",
    #"nb_etages_Appartement": "apartment_floor_number",
    "prix_metre_carre": "price_per_sqm",
    "annee_construction": "year_built",
    "specificites": "features",
    "images_urls": "images",
    "description": "description",
    "reference": "reference"
}


rename_dict_sl = {
    "id": "id",
    "legacyTracking_price": "price",
    "rawData_propertyType": "property_type",
    "mainDescription_headline": "title",
    "mainDescription_description": "description",
    "tracking_region": "region",
    "tracking_city": "city",
    "rawData_providerzipcode": "postal_code",
    "rawData_surface_main": "living_area_sqm",
    "rawData_nbroom": "num_rooms",
    "rawData_nbbedroom": "num_bedrooms",
    "energyClass": "energy_class",
    "legacyTracking_year_of_construction": "year_built",
    "gallery_images": "images",
    "hardFacts_facts": "features",
}


colonnes_cibles = list({*rename_dict_lbc.values(), *rename_dict_sl.values()})


def worker(list_csv, output_dir, rename_dict):
    """
    Traite une liste de CSV et retourne [(basename, set(final_columns)), ...]
    """
    per_file = []
    for csv_path in list_csv:
        try:
            df = pd.read_csv(csv_path, sep=";", quotechar='"')

            df.rename(columns=rename_dict, inplace=True)

            # Select les colonnes à garder
            final_cols_list = [c for c in colonnes_cibles if c in df.columns]


            df_final = df[final_cols_list]

            df_final.set_index('id', inplace=True)

            output_file = os.path.join(output_dir, os.path.basename(csv_path))
            df_final.to_csv(output_file,sep=";", index=True)

            per_file.append((os.path.basename(csv_path), set(df_final.columns.tolist())))
        except Exception as e:
            print(f"[ERROR] Erreur sur {csv_path}: {e}")
    return per_file


def inner_handler(csv_src, rename_dict, output_dir, workers, desc_prefix=""):
    output_dir = os.path.join(output_dir, desc_prefix)
    os.makedirs(output_dir, exist_ok=True)

    csv_list = glob.glob(os.path.join(csv_src, '*.csv'))

    if not csv_list:
        print(f"[WARN] Aucun fichier csv trouvé dans : {csv_src}")
        return set(), set(), {}

    # distribution équilibrée des fichiers par lot
    workers = max(1, workers)
    chunksize = math.ceil(len(csv_list) / workers)
    data = [csv_list[i:i+chunksize] for i in range(0, len(csv_list), chunksize)]
    num_processes = min(workers, len(data))

    print(f"Traitement {os.path.basename(csv_src)}: {len(csv_list)} fichiers -> {len(data)} lots, processes={num_processes}, chunksize={chunksize}")

    fun = partial(worker, rename_dict=rename_dict, output_dir=output_dir)
    all_per_file = []

    with multiprocessing.Pool(processes=num_processes) as pool:
        for chunk_result in tqdm(pool.imap_unordered(fun, data), total=len(data), desc=desc_prefix or f"Normalisation {os.path.basename(csv_src)}"):
            all_per_file.extend(chunk_result)

    per_file_final = {fname: cols for fname, cols in all_per_file}

    # calculs sûrs d'union / intersection
    final_sets = list(per_file_final.values())

    if not final_sets:
        union_cols = set()
        intersection_cols = set()
    else:
        union_cols = set().union(*final_sets)
        if len(final_sets) == 1:
            intersection_cols = set(final_sets[0])
        else:
            intersection_cols = set(final_sets[0]).intersection(*final_sets[1:])

    return union_cols, intersection_cols, per_file_final


def run(args):
    t0 = time.monotonic()

    union_sl, inter_sl, per_file_sl = inner_handler(args.sl_csv, rename_dict_sl,  args.output, args.workers, desc_prefix="SL")
    union_lbc, inter_lbc, per_file_lbc = inner_handler(args.lbc_csv, rename_dict_lbc, args.output, args.workers, desc_prefix="LBC")

    # comparaisons utiles
    always_both = inter_sl & inter_lbc
    only_in_sl = union_sl - union_lbc
    only_in_lbc = union_lbc - union_sl
    inter_of_union = union_sl & union_lbc  # apparait au moins une fois dans les deux

    if args.verbose:
        print("\nRésumé par source")
        print(f"[FILES] SL = {len(per_file_sl)} fichiers traités  |  LBC = {len(per_file_lbc)} fichiers traités")
        print()
        print(f"[SCHEMA][SL][UNION] Colonnes trouvées dans AU MOINS 1 fichier SL ({len(union_sl)}): {sorted(union_sl)}")
        print(f"[SCHEMA][SL][INTERSECTION] Colonnes présentes DANS TOUS les fichiers SL ({len(inter_sl)}): {sorted(inter_sl)}")
        print(f"[SCHEMA][SL][Naming des missing] Noms des Colonnes pouvant manquer DANS TOUS les fichiers SL ({len(union_sl-inter_sl)}): {sorted(union_sl-inter_sl)}")
        print()
        print(f"[SCHEMA][LBC][UNION] Colonnes trouvées dans AU MOINS 1 fichier LBC ({len(union_lbc)}): {sorted(union_lbc)}")
        print(f"[SCHEMA][LBC][INTERSECTION] Colonnes présentes DANS TOUS les fichiers LBC ({len(inter_lbc)}): {sorted(inter_lbc)}")
        print(f"[SCHEMA][LBC][Naming des missing] Noms des Colonnes pouvant manquer DANS TOUS les fichiers LBC ({len(union_lbc-inter_lbc)}): {sorted(union_lbc-inter_lbc)}")
        print()
        # comparaisons entre sources — libellés précis
        print(f"[COMPARE][ALWAYS_IN_ALL_FILES_BOTH] Colonnes présentes DANS TOUS les fichiers SL ET DANS TOUS les fichiers LBC: {sorted(always_both)}")
        print(f"[COMPARE][PRESENT_AT_LEAST_ONCE_BOTH] Colonnes présentes AU MOINS UNE FOIS dans SL ET AU MOINS UNE FOIS dans LBC: {sorted(inter_of_union)}")
        print()
        print(f"[COMPARE][ONLY_IN_SL] Colonnes présentes AU MOINS UNE FOIS dans SL et JAMAIS dans LBC: {sorted(only_in_sl)}")
        print(f"[COMPARE][ONLY_IN_LBC] Colonnes présentes AU MOINS UNE FOIS dans LBC et JAMAIS dans SL: {sorted(only_in_lbc)}")

    dt = time.monotonic() - t0
    print(f"\nTemps: {dt:.2f}s. CSV normalisés dans : {args.output}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sl','--sl_csv', type=str, default='../../data/SeLoger/achat/')
    parser.add_argument('-lbc','--lbc_csv', type=str, default='../../data/Lbc/achat/')
    parser.add_argument('-o','--output', type=str, default='../input/')
    parser.add_argument('-w','--workers', type=int, default=multiprocessing.cpu_count()-1)
    parser.add_argument('-v','--verbose', action='store_true')
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
