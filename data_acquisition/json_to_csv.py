import os
import argparse
import json
import glob
import csv
import time
from tqdm import tqdm

from functools import partial
import multiprocessing



def flatten_json(data_dict):
    """
    Aplatit une structure de dictionnaire/JSON.
    - Les dictionnaires imbriqués sont joints avec '_'.
    - Les listes sont converties en JSON string pour être stockées dans une seule cellule.
    """
    out = {}

    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + '_')
        elif isinstance(x, list):
            out[name[:-1]] = json.dumps(x, ensure_ascii=False)
        else:
            out[name[:-1]] = x

    flatten(data_dict)
    return out


def convert_single_json_to_csv(json_file, output_dir):
    """
    Lit un fichier JSON, aplatit chaque annonce, 
    et écrit le résultat dans un CSV avec toutes les cellules entre guillemets.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"[ERROR] Impossible de lire {json_file}: {e}")
        return None

    if not isinstance(data, list) or not data:
        print(f"[WARN] Fichier JSON vide ou non conforme : {json_file}")
        return None

    # Appliquer l'aplatissement
    records = [flatten_json(listing) for listing in data]

    # Déterminer toutes les colonnes présentes dans les enregistrements
    all_fieldnames = set()
    for record in records:
        all_fieldnames.update(record.keys())

    sorted_fieldnames = sorted(list(all_fieldnames))

    # Nom du fichier CSV
    base_name = os.path.basename(json_file)
    file_name_without_ext = os.path.splitext(base_name)[0]
    csv_filename = f"{file_name_without_ext}.csv"
    output_filepath = os.path.join(output_dir, csv_filename)

    # Écriture CSV avec toutes les cellules entre guillemets
    try:
        with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted_fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(records)
        return output_filepath
    except (IOError, csv.Error) as e:
        print(f"[ERROR] Impossible d'écrire le CSV {output_filepath}: {e}")
        return None


def run(args):
    """
    Fonction principale : parcourt les fichiers JSON et les convertit en CSV.
    """
    start_time = time.monotonic()

    input_dir = args.json
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    if not json_files:
        print(f"Aucun fichier JSON trouvé dans le dossier : {input_dir}")
        return

    workers = args.workers
    chunksize = max(1, len(json_files) // (workers * 4))
    print(f"Utilisation de {workers} workers pour convertir {len(json_files)} fichiers.")

    converter_with_output = partial(convert_single_json_to_csv, output_dir=output_dir)

    with multiprocessing.Pool(processes=workers) as pool:
        list(tqdm(pool.imap_unordered(converter_with_output, json_files, chunksize=chunksize),
                  total=len(json_files),
                  desc="Conversion des fichiers JSON en CSV"))

    end_time = time.monotonic()
    duration = end_time - start_time
    minutes, seconds = divmod(duration, 60)

    print("\n--- Statistiques Finales ---")
    print("Conversion terminée !")
    print(f"Temps total d'exécution : {int(minutes)} minutes et {seconds:.2f} secondes.")
    print(f"Les fichiers CSV sont dans le dossier : {output_dir}")



def main():
    parser = argparse.ArgumentParser(description="Conversion JSON → CSV sécurisée")
    parser.add_argument('-j', '--json', type=str, default='../../data/SeLoger/achat/json')
    parser.add_argument('-o', '--output', type=str, default='../../data/SeLoger/achat/all')
    parser.add_argument('-w', '--workers', type=int, default=max(1, multiprocessing.cpu_count()-1))
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
