import os
import glob
import argparse
import pandas as pd
import requests
from tqdm import tqdm
from functools import partial
import multiprocessing
import time
import math
from PIL import Image, ImageFile
from io import BytesIO



# Pour gérer les images tronquées qui feraient planter PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True


def format_bytes(size_bytes):
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def resize_and_pad(img: Image.Image, size=518, pad_color=(0, 0, 0)):
    """
    Redimensionne l'image en conservant le ratio et ajoute du padding pour obtenir un carré.
    Utilise LANCZOS pour une meilleure qualité de downsampling.
    """
    w, h = img.size
    
    # Si déjà à la bonne taille, on ne touche à rien
    if w == size and h == size:
        return img.convert('RGB')

    # Calcul du ratio
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Redimensionnement
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS).convert('RGB')
    
    # Création du fond
    new_img = Image.new("RGB", (size, size), pad_color)
    
    # Centrage
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    
    return new_img


def process_image_batch(batch_tasks, size=518, pad_color=(0, 0, 0), quality=80):
    """
    Traite un lot (batch) d'images.
    C'est cette fonction qui tourne sur chaque coeur CPU.
    """
    stats = {
        'downloaded': 0,
        'exists': 0,
        'error': 0,
        'bytes': 0
    }

    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) RealEstateDataset/1.0'})

    for task in batch_tasks:
        listing_id, img_idx, url, output_dir = task
        
        target_dir = os.path.join(output_dir, str(listing_id))
        filename = f"{img_idx}.jpg"
        filepath = os.path.join(target_dir, filename)

        # 1. Check si existe déjà (et non vide)
        if os.path.exists(filepath):
            if os.path.getsize(filepath) > 0:
                stats['exists'] += 1
                stats['bytes'] += os.path.getsize(filepath)
                continue
        
        # 2. Téléchargement
        try:
            # Création dossier
            os.makedirs(target_dir, exist_ok=True)

            with session.get(url, timeout=10, stream=False) as response:
                response.raise_for_status()
                
                with Image.open(BytesIO(response.content)) as im:
                    im_processed = resize_and_pad(im, size=size, pad_color=pad_color)
                    
                    # SAUVEGARDE OPTIMISÉE
                    im_processed.save(filepath, format='JPEG', quality=quality, optimize=True)
                    
                    stats['downloaded'] += 1
                    stats['bytes'] += os.path.getsize(filepath)

        except Exception as e:
            stats['error'] += 1

    session.close()
    return stats


def yield_tasks(csv_files):
    """
    Générateur qui lit tous les CSV et renvoie les images une par une.
    """
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, sep=";", usecols=['id', 'images'], dtype={'id': str, 'images': str})
            df = df.dropna(subset=['id', 'images'])

            for _, row in df.iterrows():
                listing_id = row['id']
                images_str = row['images']
                
                if not images_str: continue

                urls = images_str.split('|')
                for idx, url in enumerate(urls):
                    if len(url) > 4: 
                        yield listing_id, idx, url.strip()

        except Exception as e:
            print(f"Erreur lecture CSV {csv_path}: {e}")


def run(args):
    start_time = time.monotonic()
    
    csv_files = glob.glob(os.path.join(args.csv, "achat", "*.csv")) \
              + glob.glob(os.path.join(args.csv, "location", "*.csv"))

    if not csv_files:
        print(f"Aucun CSV trouvé dans {args.csv}")
        return

    print(f"Lecture de {len(csv_files)} fichiers CSV...")

    BATCH_SIZE = 100 
    
    task_generator = yield_tasks(csv_files)
    
    def batch_generator():
        batch = []
        for listing_id, idx, url in task_generator:
            batch.append((listing_id, idx, url, args.output))
            if len(batch) == BATCH_SIZE:
                yield batch
                batch = []
        if batch:
            yield batch

    print(f"Démarrage du téléchargement avec {args.workers} coeurs (Batch size: {BATCH_SIZE})...")
    print(f"Configuration Image : {args.size}x{args.size} pixels | Qualité JPEG : {args.quality}")
    
    worker = partial(process_image_batch, size=args.size, pad_color=args.pad_color, quality=args.quality)

    total_stats = {'downloaded': 0, 'exists': 0, 'error': 0, 'bytes': 0}
    
    with multiprocessing.Pool(processes=args.workers) as pool:
        pbar = tqdm(desc="Batchs traités", unit="batch")
        
        for batch_stats in pool.imap_unordered(worker, batch_generator()):
            total_stats['downloaded'] += batch_stats['downloaded']
            total_stats['exists'] += batch_stats['exists']
            total_stats['error'] += batch_stats['error']
            total_stats['bytes'] += batch_stats['bytes']
            pbar.update(1)
            
            pbar.set_postfix(
                dl=format_bytes(total_stats['bytes']), 
                img=total_stats['downloaded'] + total_stats['exists']
            )
            
        pbar.close()

    end_time = time.monotonic()
    minutes, seconds = divmod(end_time - start_time, 60)
    
    total_imgs = total_stats['downloaded'] + total_stats['exists'] + total_stats['error']

    print("\n" + "="*50)
    print("STATISTIQUES FINALES")
    print("="*50)
    print(f"Images Totales (vues) : {total_imgs}")
    print(f" - Téléchargées       : {total_stats['downloaded']}")
    print(f" - Déjà présentes     : {total_stats['exists']}")
    print(f" - Erreurs            : {total_stats['error']}")
    print("-" * 50)
    print(f"Taille totale disque  : {format_bytes(total_stats['bytes'])}")
    print(f"Temps d'exécution     : {int(minutes)}m {seconds:.2f}s")
    print(f"Dossier de sortie     : {args.output}")
    print("="*50)


def parse_pad_color(s):
    try:
        parts = [int(p) for p in s.split(',')]
        if len(parts) != 3: raise ValueError
        return tuple(max(0, min(255, p)) for p in parts)
    except:
        return (0,0,0)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv', type=str, default='../../output/raw_csv')
    parser.add_argument('-o', '--output', type=str, default='../../../data/images')
    parser.add_argument('-w', '--workers', type=int, default=max(1, multiprocessing.cpu_count()))
    
    # MODIFICATION SOTA : 518 est le nombre magique pour DINOv2 (14*37)
    parser.add_argument('--size', type=int, default=518, help="Taille (DINOv2 préfère 518, Standard 224/384)")
    
    # Qualité 80 suffit largement pour l'apprentissage, réduit la taille disque de 50% vs 95
    parser.add_argument('--quality', type=int, default=80, help="Qualité JPEG (1-100)")
    parser.add_argument('--pad-color', type=str, default="0,0,0", help="R,G,B")
    
    args = parser.parse_args()
    args.pad_color = parse_pad_color(args.pad_color)

    run(args)

if __name__ == '__main__':
    main()
