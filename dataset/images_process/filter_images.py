"""
Orchestrateur de Filtrage et Clustering d'Images Immobilières.

1.  **Producteurs (Workers CPU)** : Un pool de processus parallèles charge les images depuis le disque.
2.  **Consommateur (Main Process GPU)** : Le processus principal récupère les images chargées et
    applique les modèles d'IA (CLIP, DINOv3) séquentiellement.

Architecture technique :
-   Utilise `multiprocessing.Pool` pour paralléliser les I/O (lecture disque).
-   Utilise le processus principal pour l'inférence GPU (évite la duplication de la VRAM).
-   Gère les problèmes de concurrence entre `fork` et `threads` via `TOKENIZERS_PARALLELISM`.

Usage :
    python orchestrator.py --csv-root ./input --images-root ./images --results-root ./output
"""

import os
import csv
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import torch

from ai_part import init_models, process_listing_from_bytes



# -----------------------------------------------------------------------------
# WORKER : Tâches CPU (Lecture I/O)
# -----------------------------------------------------------------------------
def read_listing_images_for_listing(task):
    """
    Fonction exécutée par les workers du Pool.
    Lit toutes les images d'un dossier donné et les retourne sous forme binaire.

    Args:
        task (tuple): Un tuple contenant (listing_id, images_dir_path).

    Returns:
        tuple: (listing_id, liste_des_images)
               où liste_des_images est une liste de tuples (chemin_fichier, bytes).
    """
    listing_id, images_dir = task
    imgs = []

    # Vérification rapide de l'existence du dossier
    if not os.path.isdir(images_dir):
        return listing_id, imgs

    # Extensions acceptées
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

    try:
        # Parcours récursif ou simple du dossier
        for root, _, files in os.walk(images_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    p = os.path.join(root, f)
                    try:
                        # Lecture binaire du fichier
                        with open(p, "rb") as fh:
                            b = fh.read()
                        imgs.append((p, b))
                    except Exception:
                        pass
    except Exception:
        pass

    return listing_id, imgs


# -----------------------------------------------------------------------------
# ORCHESTRATEUR : Gestion du Pipeline
# -----------------------------------------------------------------------------
def process_all(csv_root, images_root, results_root, args):
    """
    Fonction principale orchestrant le chargement et le traitement IA.
    MODIFICATION: Sauvegarde directement dans results_root/listing_id.
    """

    # 1. Collecte des fichiers CSV
    csv_files = sorted(list(Path(csv_root, "achat").glob("*.csv")) + 
                       list(Path(csv_root, "location").glob("*.csv")))

    if not csv_files:
        print("Avertissement : Aucun fichier CSV trouvé.")
        return

    # S'assurer que le dossier racine de sortie existe
    root_out = Path(results_root)
    root_out.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Initialisation du Pool AVANT les modèles
    # -------------------------------------------------------------------------
    n_workers = args.num_workers
    print(f"--- [SYSTÈME] Démarrage du Pool I/O avec {n_workers} workers ---")
    pool = Pool(processes=n_workers)

    try:
        # ---------------------------------------------------------------------
        # ÉTAPE IA : Chargement des modèles sur le Main Process
        # ---------------------------------------------------------------------
        print("--- [GPU] Chargement des modèles IA ---")
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        clip_cfg, model_vis, processor_vis = init_models(
            device=device,
            clip_hf_id=args.clip_hf_model,
            visual_id=args.visual_model
        )
        print(f"--- [GPU] Modèles chargés sur {device}. ---")

        check_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

        # Boucle sur chaque fichier CSV trouvé
        for csv_path in csv_files:
            csv_base = Path(csv_path).stem 
            print(f"\n=== Traitement du CSV : {csv_path.name} ===")

            # Lecture du CSV
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                rows = list(reader)

            tasks = []          
            listing_meta = {}   
            report_rows = []    

            # Préparation des tâches
            for row in rows:
                listing_id = str(row.get('id', '')).strip()
                if not listing_id: 
                    continue

                images_dir = os.path.join(images_root, listing_id)
                
                # --- MODIFICATION ICI ---
                # On pointe directement vers output/ID au lieu de output/csv/ID
                result_dir = str(root_out / listing_id)
                # ------------------------

                try:
                    num_rooms = int(row.get('num_rooms'))
                except ValueError:
                    num_rooms = 4

                # Logique de reprise (Skip si déjà fait)
                already_done = False
                if os.path.isdir(result_dir):
                    if any(os.path.splitext(f)[1].lower() in check_exts for f in os.listdir(result_dir)):
                        already_done = True

                if already_done:
                    report_rows.append({
                        "listing_id": listing_id, "total": 0, "kept": 0, "selected": 0, 
                        "clusters": 0, "selected_paths": "SKIPPED_ALREADY_DONE", "time_s": 0
                    })
                    continue

                tasks.append((listing_id, images_dir))
                listing_meta[listing_id] = {"result_dir": result_dir, "num_rooms": num_rooms}

            print(f"Stats : {len(rows)} total | {len(tasks)} à traiter | {len(report_rows)} ignorés")

            if tasks:
                it = pool.imap_unordered(read_listing_images_for_listing, tasks, chunksize=1)
                pbar = tqdm(total=len(tasks), desc=f"Traitement {csv_base}")

                for listing_id, images_bytes in it:
                    meta = listing_meta.get(listing_id)
                    if not meta: 
                        continue

                    try:
                        summary = process_listing_from_bytes(
                            listing_id=listing_id,
                            images_bytes=images_bytes,
                            results_dir=meta["result_dir"],
                            clip_cfg=clip_cfg,
                            model_vis=model_vis,
                            processor_vis=processor_vis,
                            device=device,
                            batch_size=args.batch_size,
                            num_rooms=meta["num_rooms"],
                            max_pca_components=args.max_pca_components,
                            clip_debug=args.clip_debug
                        )
                    except Exception as e:
                        print(f"Erreur critique listing {listing_id}: {e}")
                        summary = {"listing_id": listing_id, "total": 0, "kept": 0, "selected": 0, "clusters": 0, "selected_paths": []}

                    # Calcul du chemin relatif par rapport à la racine (args.results_root)
                    rel_selected = []
                    for p in summary.get("selected_paths", []):
                        try: 
                            rel_selected.append(os.path.relpath(p, root_out))
                        except ValueError: 
                            rel_selected.append(p)

                    report_rows.append({
                        "listing_id": listing_id,
                        "total": summary.get("total", 0),
                        "kept": summary.get("kept", 0),
                        "selected": summary.get("selected", 0),
                        "clusters": summary.get("clusters", 0),
                        "selected_paths": "|".join(rel_selected),
                        "time_s": summary.get("time_s", 0.0)
                    })

                    pbar.update(1)

                pbar.close()

            # Sauvegarde à la racine avec le nom du CSV source pour éviter les collisions
            report_name = f"report_{csv_base}.csv"
            report_path = root_out / report_name

            with open(report_path, "w", newline='', encoding='utf-8') as rf:
                fieldnames = ["listing_id", "total", "kept", "selected", "clusters", "selected_paths", "time_s"]
                writer = csv.DictWriter(rf, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(report_rows)
            print(f"Rapport sauvegardé : {report_path}")

    finally:
        print("\n--- [SYSTÈME] Fermeture du Pool de workers ---")
        pool.close()
        pool.join()


# -----------------------------------------------------------------------------
# POINT D'ENTRÉE CLI
# -----------------------------------------------------------------------------
def main():
    """Configuration des arguments en ligne de commande et lancement."""
    parser = argparse.ArgumentParser(description="Pipeline de filtrage d'images immo (CPU I/O + GPU AI)")
    
    # Chemins
    parser.add_argument('-c','--csv-root', type=str, default='../../output/raw_csv', help="Dossier contenant les CSV (achat/location)")
    parser.add_argument('-i','--images-root', type=str, default='../../../data/images', help="Dossier source des images")
    parser.add_argument('-r','--results-root', type=str, default='../../output/filtered_images', help="Dossier de sortie")

    # Modèles
    parser.add_argument('--visual-model', type=str, default='facebook/dinov3-vits16-pretrain-lvd1689m', help="HuggingFace ID pour DINO")
    parser.add_argument('--clip-hf-model', type=str, default='openai/clip-vit-base-patch32', help="HuggingFace ID pour CLIP")

    # Hyperparamètres
    parser.add_argument('--batch-size', type=int, default=64, help="Taille du batch pour l'inférence")
    parser.add_argument('--max-pca-components', type=int, default=16, help="Composantes PCA pour le clustering")

    # Système
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'], help="Périphérique de calcul")
    parser.add_argument('--num-workers', type=int, default=max(cpu_count(), 1), help="Nombre de processus CPU pour la lecture des images")
    parser.add_argument('--clip-debug', action='store_true', help="Active les logs détaillés pour CLIP")

    args = parser.parse_args()

    print("=== DÉMARRAGE DE L'ORCHESTRATEUR ===")
    print(f"Config : Device={args.device} | Workers={args.num_workers} | Batch={args.batch_size}")

    process_all(args.csv_root, args.images_root, args.results_root, args)

    print("=== TRAITEMENT TERMINÉ ===")

if __name__ == "__main__":
    main()
