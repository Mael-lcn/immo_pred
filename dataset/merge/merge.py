import os
import argparse
import pandas as pd
import time
import glob
from tqdm import tqdm

from functools import partial
import multiprocessing
import csv



# Tmp à cause de nom des fils....
import sys
sys.stdout.reconfigure(encoding='utf-8')



colonnes_cibles = [
    "id", "price", "property_type", "title", "description", "region", "city", "postal_code","living_area", "bedrooms", "rooms", "floor","energy_class",
    "year_built", "sale_type", "seller_type","reference", "images",
]

rename_map = {
    "prix": "price",
    "type_bien": "property_type",
    "titre": "title",
    "description": "description",
    "ville": "city",
    "codePostal": "postal_code",
    "surface_habitable": "living_area",
    "nb_chambres": "bedrooms",
    "nb_pieces": "rooms",
    "nb_etages_Immeuble": "floor",
    "classe_energetique": "energy_class",
    "annee_construction": "year_built",
    "type_vente": "sale_type",
    "type_vendeur": "seller_type",
    "reference": "reference",
    "images_urls": "images",
}


def worker(list_csv, output_dir):
    """
    Parcourt une liste de fichiers CSV (chunk), les lit avec pandas,
    renomme les colonnes, sauvegarde et (si verbose) retourne la liste
    des couples (basename, set(columns)) pour chaque CSV traité.
    """
    per_file = []

    for csv_path in list_csv:
        try:
            # Lecture
            df = pd.read_csv(csv_path, sep=";", dtype=str, encoding="utf8", quoting=csv.QUOTE_NONE)

            # Renommage
            df.rename(columns=rename_map, inplace=True)

            # Sauvegarde finale
            output_file = os.path.join(output_dir, os.path.basename(csv_path))
            df.to_csv(output_file, index=False)

            per_file.append((os.path.basename(csv_path), set(df.columns)))

        except Exception as e:
            print(f"[ERROR] Erreur sur {csv_path}: {e}")

    # Retourne la liste des (filename, set(columns)) pour ce chunk
    return per_file


def run(args):
    """
    Orchestration principale.
    inner_handler -> traite un dossier et retourne (not_always, always, per_file_cols)
    """
    start_time = time.monotonic()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # inner_handler traite un dossier csv_src et retourne le tuple demandé
    def inner_handler(csv_src):
        csv_list = glob.glob(os.path.join(csv_src, '*.csv'))

        if not csv_list:
            print(f"[WARN] Aucun fichier csv trouvé dans le dossier : {csv_src}")
            return (set(), set(), {})  # tuple vide mais cohérent

        # calcul du chunksize pour diviser la liste par worker
        if args.workers > 0:
            chunksize = max(1, len(csv_list) // (args.workers))
        else:
            chunksize = len(csv_list)

        data = [csv_list[i:i+chunksize] for i in range(0, len(csv_list), chunksize)]

        print(f"Utilisation de {args.workers} workers avec une taille de lot (chunksize) de {chunksize} (dossiers: {len(csv_list)} fichiers).")

        fun = partial(worker, output_dir=output_dir)
        all_per_file = []

        # multiprocessing
        with multiprocessing.Pool(processes=args.workers) as pool:
            # imap_unordered renvoie un résultat par chunk
            for chunk_result in tqdm(pool.imap_unordered(fun, data), total=len(data), desc=f"Normalisation des CSV dans {os.path.basename(csv_src)}"):
                all_per_file.extend(chunk_result)

        # Construire dict filename -> set(columns)
        per_file_cols = {fname: cols for fname, cols in all_per_file}

        # calculs d'union et d'intersection
        sets = list(per_file_cols.values())
        union_cols = set().union(*sets)
        intersection_cols = set(sets[0]).intersection(*sets[1:]) if len(sets) > 1 else set(sets[0])

        not_always = union_cols - intersection_cols
        always = intersection_cols

        # Affichage synthétique (interne) si verbose
        if args.verbose:
            print(f"\n[Dossier] {csv_src}: {len(per_file_cols)} fichiers analysés.")
            print(f"- Colonnes totales rencontrées (union) : {len(union_cols)}")
            print(f"- Colonnes présentes dans tous les fichiers (intersection) : {len(always)}")
            print(f"- Colonnes manquantes dans au moins un fichier : {len(not_always)}\n")

        return (not_always, always, per_file_cols)


    # Appels à inner_handler pour les deux dossiers (SL et LBC)
    result_sl = inner_handler(args.sl_csv)
    #result_lbc = inner_handler(args.lbc_csv) if args.lbc_csv else (set(), set(), {})
    result_lbc = (set(), set(), {})

    if args.verbose:
        # Déballage des résultats
        not_always_sl, always_sl, per_file_sl = result_sl
        not_always_lbc, always_lbc, per_file_lbc = result_lbc

        # Comparaisons inter-dossiers
        union_sl = set().union(*per_file_sl.values()) if per_file_sl else set()
        union_lbc = set().union(*per_file_lbc.values()) if per_file_lbc else set()

        # Colonnes toujours présentes dans A et B -> intersection
        always_both = always_sl & always_lbc

        # Colonnes toujours dans A mais pas toujours dans B (et inverse)
        always_only_sl = always_sl - always_lbc
        always_only_lbc = always_lbc - always_sl

        # Colonnes présentes dans A (au moins une fois) mais jamais dans B (et inverse)
        only_in_sl = union_sl - union_lbc
        only_in_lbc = union_lbc - union_sl

        # Impression finale
        print("\n--- Comparaison Finale entre dossiers ---")
        print(f"Fichiers analysés : SL={len(per_file_sl)} | LBC={len(per_file_lbc)}")
        print("\nColonnes toujours présentes (intersection SL ∩ LBC):")
        print(sorted(always_both))

        print("\nColonnes toujours dans SL mais pas toujours dans LBC:")
        print(sorted(always_only_sl))

        print("\nColonnes toujours dans LBC mais pas toujours dans SL:")
        print(sorted(always_only_lbc))

        print("\nColonnes présentes dans SL (au moins une fois) mais jamais dans LBC:")
        print(sorted(only_in_sl))

        print("\nColonnes présentes dans LBC (au moins une fois) mais jamais dans SL:")
        print(sorted(only_in_lbc))

        print("\n--- Détails par dossier ---")
        print("SL - colonnes toujours présentes:", sorted(always_sl))
        print("SL - colonnes pas toujours présentes:", sorted(not_always_sl))
        print("LBC - colonnes toujours présentes:", sorted(always_lbc))
        print("LBC - colonnes pas toujours présentes:", sorted(not_always_lbc))

        """
        # Exemple de sortie par fichier (peu verbeux si beaucoup de fichiers)
        print("\nExtrait des colonnes par fichier (quelques fichiers si nombreux):")
        for i, (fname, cols) in enumerate(list(per_file_sl.items())[:10]):
            print(f" SL/{fname}: {sorted(cols)}")
        for i, (fname, cols) in enumerate(list(per_file_lbc.items())[:10]):
            print(f" LBC/{fname}: {sorted(cols)}")
        """

    # Statistiques temps
    end_time = time.monotonic()
    duration = end_time - start_time
    minutes, seconds = divmod(duration, 60)

    print("\n--- Statistiques Finales ---")
    print(f"Temps total d'exécution : {int(minutes)} minutes et {seconds:.2f} secondes.")
    print(f"Les CSV normalisés sont dans : {output_dir}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sl', '--sl_csv', type=str, default='../data/leBonCoinAchats')
    parser.add_argument('-lbc', '--lbc_csv', type=str, default='../data/seLogerAchats')
    parser.add_argument('-o', '--output', type=str, default='../output/SeLoger/')
    parser.add_argument('-w', '--workers', type=int, default=multiprocessing.cpu_count()-1)
    parser.add_argument('-v', '--verbose', action='store_true', help="Affiche les schémas/colonnes par fichier et les comparaisons détaillées.")
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
