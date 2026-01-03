"""
Orchestrateur de Filtrage et Clustering d'Images Immobilières.

Ce script gère le parallélisme pour traiter massivement des annonces immobilières.
- Lecture I/O sur CPU (Multiprocessing).
- Inférence IA sur GPU (Processus Principal).
- Clustering dynamique basé sur la similarité visuelle (SOTA).

Usage :
    python filter_images.py --csv-root ./input --images-root ./images --results-root ./output
"""

import os
import csv
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import torch

# Import de la logique IA refondue
from ai_part import init_models, process_listing_smart_clustering



# -----------------------------------------------------------------------------
# WORKER : Tâches CPU (Lecture I/O Optimisée)
# -----------------------------------------------------------------------------
def read_listing_images_for_listing(task):
    """
    Lit les images d'un dossier.
    Optimisé avec os.scandir pour éviter les appels systèmes lents.
    """
    listing_id, images_dir = task
    imgs = []

    if not os.path.isdir(images_dir):
        return listing_id, imgs

    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

    try:
        with os.scandir(images_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    # Check extension rapide (minuscule)
                    if os.path.splitext(entry.name)[1].lower() in valid_exts:
                        try:
                            # Lecture binaire pure (rapide)
                            with open(entry.path, "rb") as fh:
                                b = fh.read()
                            # Filtre basique : fichier vide ou corrompu (trop petit)
                            if len(b) > 512: 
                                imgs.append((entry.path, b))
                        except Exception:
                            pass
    except Exception:
        pass

    return listing_id, imgs


# -----------------------------------------------------------------------------
# ORCHESTRATEUR : Gestion du Pipeline
# -----------------------------------------------------------------------------
def process_all(csv_root, images_root, results_root, args):
    # 1. Recherche des CSV (Achat et Location)
    csv_root_path = Path(csv_root)
    csv_files = sorted(list(csv_root_path.glob("**/achat/*.csv")) + 
                       list(csv_root_path.glob("**/location/*.csv")))

    if not csv_files:
        print(f"[WARN] Aucun fichier CSV trouvé dans {csv_root}")
        return

    # Création racine sortie
    root_out = Path(results_root)
    root_out.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. CHARGEMENT MODÈLES (Une seule fois, sur le GPU Main Process)
    # -------------------------------------------------------------------------
    print("--- [GPU] Chargement des modèles IA ---")
    
    # Initialisation centralisée
    try:
        models_context = init_models(
            device=args.device,
            clip_hf_id=args.clip_hf_model,
            visual_id=args.visual_model
        )
    except Exception as e:
        print(f"[FATAL] Impossible d'initialiser les modèles : {e}")
        return

    print("--- [GPU] Modèles prêts. ---")

    # -------------------------------------------------------------------------
    # 3. DÉMARRAGE DU POOL I/O
    # -------------------------------------------------------------------------
    # On limite le nombre de workers I/O. Trop de workers saturent la RAM 
    # car ils envoient tous des images binaires au processus principal.
    n_workers = min(args.num_workers, 8)
    print(f"--- [CPU] Démarrage du Pool I/O avec {n_workers} workers ---")

    pool = Pool(processes=n_workers)

    try:
        # Boucle sur chaque fichier CSV
        for csv_path in csv_files:
            print(f"\n=== Traitement CSV : {csv_path.name} ===")

            # Lecture et Préparation des tâches
            tasks = []
            listing_meta = {} # Pour garder le lien ID -> Dossier de sortie

            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter=';')
                    for row in reader:
                        lid = row.get('id', '').strip()
                        if not lid: continue

                        # Dossier source des images
                        img_dir = Path(images_root) / lid
                        
                        # Dossier destination
                        out_dir = root_out / lid

                        # SKIP INTELLIGENT : Si le dossier de sortie existe et n'est pas vide
                        if out_dir.exists() and any(out_dir.iterdir()):
                            continue

                        # On n'ajoute la tâche que si on doit la traiter
                        tasks.append((lid, str(img_dir)))
                        listing_meta[lid] = {"out_dir": str(out_dir)}
            except Exception as e:
                print(f"[ERREUR] Lecture CSV {csv_path}: {e}")
                continue

            if not tasks:
                print("   -> Aucune nouvelle annonce à traiter.")
                continue

            print(f"   -> {len(tasks)} annonces en file d'attente.")

            # Lancement du Pipeline : 
            # Workers lisent le disque -> Main Process fait l'IA
            iterator = pool.imap_unordered(read_listing_images_for_listing, tasks, chunksize=4)
            
            report_rows = []

            for listing_id, images_bytes in tqdm(iterator, total=len(tasks), desc="Processing"):
                # Si le worker n'a rien trouvé (dossier vide ou inexistant)
                if not images_bytes:
                    continue

                meta = listing_meta.get(listing_id)
                if not meta: continue

                try:
                    # APPEL DE LA FONCTION IA OPTIMISÉE
                    summary = process_listing_smart_clustering(
                        listing_id=listing_id,
                        images_bytes=images_bytes,
                        results_dir=meta["out_dir"],
                        models_context=models_context,
                        device=args.device,
                        similarity_threshold=args.similarity_threshold,
                        min_images=args.min_images,
                        max_images=args.max_images
                    )
                    report_rows.append(summary)

                except Exception as e:
                    print(f"\n[ERREUR TRAITEMENT] ID {listing_id}: {e}")

            # Sauvegarde d'un rapport par CSV pour monitoring
            if report_rows:
                report_file = root_out / f"report_{csv_path.stem}.csv"
                try:
                    keys = report_rows[0].keys()
                    with open(report_file, 'w', newline='', encoding='utf-8') as f:
                        dw = csv.DictWriter(f, fieldnames=keys)
                        dw.writeheader()
                        dw.writerows(report_rows)
                except Exception as e:
                    print(f"[ERREUR] Ecriture rapport : {e}")

    except KeyboardInterrupt:
        print("\n[STOP] Interruption utilisateur...")
    finally:
        print("\n--- [SYSTÈME] Fermeture du Pool ---")
        pool.close()
        pool.join()
        print("=== TRAITEMENT TERMINÉ ===")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline SOTA de filtrage d'images immo")
    
    # Chemins
    parser.add_argument('--csv-root', type=str, default='../../output/raw_csv')
    parser.add_argument('--images-root', type=str, default='../../../data/images')
    parser.add_argument('--results-root', type=str, default='../../output/filtered_images')
    
    # Modèles
    # DINOv2 Base est le meilleur compromis Vitesse/Performance pour le clustering géométrique
    parser.add_argument('--visual-model', type=str, default='facebook/dinov2-base') 
    parser.add_argument('--clip-hf-model', type=str, default='openai/clip-vit-base-patch32')

    # Paramètres de Clustering (SOTA)
    parser.add_argument('--similarity-threshold', type=float, default=0.25, 
                        help="Seuil de distance Cosine (0.0-1.0). 0.25 = Très strict (images quasi identiques). 0.40 = Plus large.")
    parser.add_argument('--min-images', type=int, default=3, help="Minimum d'images à conserver par annonce (si dispo)")
    parser.add_argument('--max-images', type=int, default=15, help="Maximum d'images à conserver par annonce")

    # Hardware
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num-workers', type=int, default=max(cpu_count()-2, 1))

    args = parser.parse_args()

    # Fix pour le multiprocessing PyTorch sous Linux/Windows
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    process_all(args.csv_root, args.images_root, args.results_root, args)
