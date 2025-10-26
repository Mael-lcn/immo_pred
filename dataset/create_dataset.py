import os
import argparse
import json
import glob
import csv
import multiprocessing
import time
from tqdm import tqdm
from functools import partial



def run(args):
    """
    Fonction principale qui drive la lecture des JSON et la conversion en CSV.
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json', type=str, default='../../data/SeLoger/achat/json', help="Dossier contenant les fichiers JSON source.")
    parser.add_argument('-o', '--output', type=str, default='../../data/SeLoger/achat/csv', help="Dossier de sortie pour les fichiers CSV complets générés.")
    parser.add_argument('-w', '--workers', type=int, default=multiprocessing.cpu_count()-1)
    args = parser.parse_args()

    run(args)

if __name__ == '__main__':
    main()
