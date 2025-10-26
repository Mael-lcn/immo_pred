import os
import argparse
import json
import glob
import requests
from tqdm import tqdm
from functools import partial
import multiprocessing
import time
import math



def format_bytes(size_bytes):
    """Convertit une taille en octets en un format lisible (KB, MB, GB...)."""
    if size_bytes == 0:
        return "0 B"

    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)

    return f"{s} {size_name[i]}"


def download_image(image_info, output_dir):
    """
    Télécharge une seule image et retourne sa taille en octets.
    Retourne 0 en cas d'échec, ou la taille du fichier s'il existe déjà.
    """
    listing_id, image_key, image_url = image_info

    target_dir = os.path.join(output_dir, listing_id)
    os.makedirs(target_dir, exist_ok=True)

    filepath = os.path.join(target_dir, f"{image_key}.jpg")

    # Si le fichier existe déjà, on retourne sa taille sans le retélécharger
    if os.path.exists(filepath):
        return os.path.getsize(filepath)

    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
        
        image_data = response.content
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        # Retourne la taille de l'image téléchargée
        return len(image_data)
    except requests.exceptions.RequestException:
        print(f"Warning: in {image_url}")
        return 0


def run(args):
    """
    Fonction principale qui orchestre la lecture des JSON et le téléchargement.
    """
    start_time = time.monotonic()
    
    input_dir = args.json
    output_dir = args.output
    
    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    
    if not json_files:
        print(f"Aucun fichier JSON trouvé dans le dossier : {input_dir}")
        return

    all_image_tasks = []
    for file_path in tqdm(json_files, desc="Analyse des fichiers JSON"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)

            for listing in batch_data:
                listing_id = listing.get('id')
                images = listing.get('gallery', {}).get('images', [])
                
                if not listing_id or not images:
                    continue
                    
                for image in images:
                    if image.get('key') and image.get('url'):
                        all_image_tasks.append((listing_id, image['key'], image['url']))
        except (json.JSONDecodeError, IOError) as e:
            print(f"\nErreur en lisant le fichier {file_path}: {e}")

    if not all_image_tasks:
        print("Aucune image à télécharger n'a été trouvée dans les fichiers JSON.")
        return

    nb_aller_retour = 16
    if args.workers > 0:
        chunksize = max(1, len(all_image_tasks) // (args.workers * nb_aller_retour))
    else:
        chunksize = 1

    print(f"Utilisation de {args.workers} workers avec une taille de lot (chunksize) de {chunksize}.")

    downloader = partial(download_image, output_dir=output_dir)

    with multiprocessing.Pool(processes=args.workers) as pool:
        # 'results' contiendra maintenant la taille de chaque image (ou 0 en cas d'échec)
        results = list(tqdm(pool.imap_unordered(downloader, all_image_tasks, chunksize=chunksize), 
                            total=len(all_image_tasks), 
                            desc="Téléchargement des images"))

    # Calcule et affiche les statistiques finales ---
    total_size_bytes = sum(results)
    success_count = sum(1 for size in results if size > 0)
    end_time = time.monotonic()
    duration = end_time - start_time
    minutes, seconds = divmod(duration, 60)

    print("\n--- Statistiques Finales ---")
    print(f"Téléchargement terminé ! {success_count}/{len(all_image_tasks)} images téléchargées.")
    print(f"Taille totale des images sur le disque : {format_bytes(total_size_bytes)}.")
    print(f"Temps total d'exécution : {int(minutes)} minutes et {seconds:.2f} secondes.")
    print(f"Les images sont dans le dossier : {output_dir}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json', type=str, default='../../data/SeLoger/Toulouse/json', help="Dossier contenant les fichiers JSON.")
    parser.add_argument('-o', '--output', type=str, default='../../data/SeLoger/Toulouse/images', help="Dossier de sortie pour les images.")
    parser.add_argument('-w', '--workers', type=int, default=multiprocessing.cpu_count()-1, help="Nombre de processus parallèles (par défaut: tous les cœurs disponibles).")
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
