import os, glob
import argparse
import pandas as pd
import requests
from tqdm import tqdm
from functools import partial
import multiprocessing
import time
import math
from PIL import Image
from io import BytesIO



def format_bytes(size_bytes):
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def resize_and_pad(img: Image.Image, size=672, pad_color=(0,0,0)):
    w, h = img.size
    if w == size and h == size:
        return img.convert('RGB')
    scale = size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_resized = img.resize((new_w, new_h), Image.BICUBIC).convert('RGB')
    new_img = Image.new("RGB", (size, size), pad_color)
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    return new_img


def download_image(image_info, output_dir, size=672, pad_color=(0,0,0)):
    """
    Retourne (status, size_bytes) où status est 'exists', 'downloaded' ou 'error'.
    """
    listing_id, image_key, image_url = image_info
    target_dir = os.path.join(output_dir, listing_id)
    os.makedirs(target_dir, exist_ok=True)
    filepath = os.path.join(target_dir, f"{image_key}.jpg")

    # si existe, on considère déjà téléchargé / processe -> skip
    if os.path.exists(filepath):
        try:
            return ('exists', os.path.getsize(filepath))
        except Exception:
            return ('exists', 0)

    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
        image_data = response.content

        # ouvrir directement depuis la mémoire
        try:
            with Image.open(BytesIO(image_data)) as im:
                im = im.convert('RGB')
                im2 = resize_and_pad(im, size=size, pad_color=pad_color)
                im2.save(filepath, format='JPEG', quality=95, optimize=True)
                return ('downloaded', os.path.getsize(filepath))
        except Exception as e:
            print(f"Warning: impossible de traiter {image_url} ({e})")
            return ('error', 0)

    except requests.exceptions.RequestException:
        print(f"Warning: impossible de télécharger {image_url}")
        return ('error', 0)


def process_csv(csv_path, output_dir, size=672, pad_color=(0,0,0)):
    """
    Retourne :
    (total_size_bytes, downloaded_count, requested_count, failed_count, exists_count)
    """
    total_size = 0
    downloaded_count = 0
    requested_count = 0
    failed_count = 0
    exists_count = 0

    try:
        df = pd.read_csv(csv_path, sep=";", usecols=['id', 'images'])

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"CSV {os.path.basename(csv_path)}"):
            listing_id = row.get('id')
            images_str = row.get('images')
            if not listing_id or pd.isna(images_str):
                continue

            urls = images_str.split('|')
            for idx, url in enumerate(urls):
                requested_count += 1
                status, size_bytes = download_image((str(listing_id), str(idx), url), output_dir, size=size, pad_color=pad_color)

                if status == 'downloaded':
                    downloaded_count += 1
                    total_size += size_bytes
                elif status == 'exists':
                    exists_count += 1
                    # on peut ajouter la taille existante si souhaité :
                    total_size += size_bytes
                elif status == 'error':
                    failed_count += 1

    except Exception as e:
        print(f"Erreur lors du traitement du CSV {csv_path}: {e}")

    return total_size, downloaded_count, requested_count, failed_count, exists_count


def run(args):
    start_time = time.monotonic()
    input_dir = args.csv
    output_dir = args.output
    size = args.size
    pad_color = args.pad_color  # tuple (R,G,B)

    csv_files = glob.glob(os.path.join(args.csv, "achat", "*.csv")) \
           + glob.glob(os.path.join(args.csv, "location", "*.csv"))

    if not csv_files:
        print(f"Aucun CSV trouvé dans {input_dir}")
        return

    print(f"{len(csv_files)} fichiers CSV trouvés, utilisation de {args.workers} workers.")
    worker_fn = partial(process_csv, output_dir=output_dir, size=size, pad_color=pad_color)

    with multiprocessing.Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(worker_fn, csv_files),
                            total=len(csv_files),
                            desc="Téléchargement des images"))

    # résultats : list de tuples (total_size, downloaded_count, requested_count, failed_count, exists_count)
    total_size_bytes = sum(r[0] for r in results)
    total_downloaded = sum(r[1] for r in results)
    total_requested = sum(r[2] for r in results)
    total_failed = sum(r[3] for r in results)
    total_exists = sum(r[4] for r in results)

    end_time = time.monotonic()
    minutes, seconds = divmod(end_time - start_time, 60)

    print("\n--- Statistiques Finales ---")
    print(f"Images demandées     : {total_requested}")
    print(f"Images téléchargées  : {total_downloaded}")
    print(f"Images déjà présentes : {total_exists}")
    print(f"Échecs (non téléchargées) : {total_failed}")
    print(f"{total_downloaded + total_exists}/{total_requested} images présentes ou téléchargées.")
    print(f"Taille totale des images sur le disque : {format_bytes(total_size_bytes)}.")
    print(f"Temps total d'exécution : {int(minutes)} minutes et {seconds:.2f} secondes.")
    print(f"Les images sont dans le dossier : {output_dir}")


def parse_pad_color(s):
    try:
        parts = [int(p) for p in s.split(',')]
        if len(parts) != 3:
            raise ValueError
        return tuple(max(0, min(255, p)) for p in parts)
    except Exception:
        print("pad-color invalide, utilisation de (0,0,0)")
        return (0,0,0)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv', type=str, default='../../output/raw_csv')
    parser.add_argument('-o', '--output', type=str, default='../../../data/images')
    parser.add_argument('-w', '--workers', type=int, default=max(1, multiprocessing.cpu_count()-1))
    parser.add_argument('--size', type=int, default=672, help="taille finale (carre) ex: 672")
    parser.add_argument('--pad-color', type=str, default="0,0,0", help="couleur du padding en R,G,B (ex: 255,255,255)")
    args = parser.parse_args()

    args.pad_color = parse_pad_color(args.pad_color)

    run(args)


if __name__ == '__main__':
    main()

