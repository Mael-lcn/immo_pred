import os
import argparse
import pandas as pd
import requests
from tqdm import tqdm
from functools import partial
import multiprocessing
import time
import math
import re



def format_bytes(size_bytes):
    if size_bytes == 0:
        return "0 B"

    size_name = ("B", "KB", "MB", "GB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def download_image(image_info, output_dir):
    listing_id, image_key, image_url = image_info
    target_dir = os.path.join(output_dir, listing_id)

    os.makedirs(target_dir, exist_ok=True)
    filepath = os.path.join(target_dir, f"{image_key}.jpg")

    if os.path.exists(filepath):
        return os.path.getsize(filepath)

    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
        image_data = response.content
        with open(filepath, 'wb') as f:
            f.write(image_data)
        return len(image_data)
    except requests.exceptions.RequestException:
        print(f"Warning: impossible de télécharger {image_url}")
        return 0


def process_csv(csv_path, output_dir):
    total_size = 0
    success_count = 0
    image_count = 0

    try:
        df = pd.read_csv(csv_path, sep=";", usecols=['id', 'images'])
    
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"CSV {os.path.basename(csv_path)}"):
            listing_id = row.get('id')
            images_str = row.get('images')
            if not listing_id or pd.isna(images_str):
                continue

            # on récupère toutes les URLs dans la chaîne séparée par "|"
            urls = images_str.split('|')
            for idx, url in enumerate(urls):
                size = download_image((str(listing_id), str(idx), url), output_dir)
                total_size += size
                if size > 0:
                    success_count += 1
                image_count += 1

    except Exception as e:
        print(f"Erreur lors du traitement du CSV {csv_path}: {e}")

    return total_size, success_count, image_count



def run(args):
    start_time = time.monotonic()
    input_dir = args.csv
    output_dir = args.output

    csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"Aucun CSV trouvé dans {input_dir}")
        return

    print(f"{len(csv_files)} fichiers CSV trouvés, utilisation de {args.workers} workers.")

    with multiprocessing.Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(partial(process_csv, output_dir=output_dir), csv_files),
                            total=len(csv_files),
                            desc="Téléchargement des images"))

    total_size_bytes = sum(r[0] for r in results)
    total_success = sum(r[1] for r in results)
    total_images = sum(r[2] for r in results)
    end_time = time.monotonic()
    minutes, seconds = divmod(end_time - start_time, 60)

    print("\n--- Statistiques Finales ---")
    print(f"{total_success}/{total_images} images téléchargées avec succès.")
    print(f"Taille totale des images sur le disque : {format_bytes(total_size_bytes)}.")
    print(f"Temps total d'exécution : {int(minutes)} minutes et {seconds:.2f} secondes.")
    print(f"Les images sont dans le dossier : {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv', type=str, default='../output/csv')
    parser.add_argument('-o', '--output', type=str, default='../output/images')
    parser.add_argument('-w', '--workers', type=int, default=max(1, multiprocessing.cpu_count()-1))
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
