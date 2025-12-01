import os
import csv
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import torch

from ai_part import init_models, process_listing_from_bytes

# -------------------------
# worker: CPU-bound I/O reading images for a listing
# returns list of (path, bytes)
# -------------------------
def read_listing_images_for_listing(task):
    """
    task = (listing_id, images_dir)
    returns (listing_id, [(path, bytes), ...])
    """
    listing_id, images_dir = task
    imgs = []
    if not os.path.isdir(images_dir):
        return listing_id, imgs
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    try:
        for root, _, files in os.walk(images_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    p = os.path.join(root, f)
                    try:
                        with open(p, "rb") as fh:
                            b = fh.read()
                        imgs.append((p, b))
                    except Exception as e:
                        print(f"[worker:{listing_id}] read fail {p}: {e}")
    except Exception as e:
        print(f"[worker:{listing_id}] os.walk fail: {e}")
    return listing_id, imgs


# -------------------------
# orchestrateur:
# -------------------------
def process_all(csv_root, images_root, results_root, args):
    # collect CSV files
    csv_files = sorted(list(Path(csv_root, "achat").glob("*.csv")) + list(Path(csv_root, "location").glob("*.csv")))
    if not csv_files:
        print("Aucun CSV trouvé.")
        return

    # init models
    device = args.device if args.device != "auto" else ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    clip_cfg, model_vis, processor_vis = init_models(device=device,
                                                    clip_hf_id=args.clip_hf_model,
                                                    visual_id=args.visual_model)

    # create pool for I/O tasks (num_workers ~ cpu_count - 1)
    n_workers = max(1, min(args.num_workers, cpu_count()-1))
    pool = Pool(processes=n_workers)

    # Extensions d'images pour la vérification
    check_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

    for csv_path in csv_files:
        csv_base = Path(csv_path).stem
        print(f"\n=== Processing CSV: {csv_path} -> bucket: {csv_base} ===")
        results_csv_dir = Path(results_root) / csv_base
        results_csv_dir.mkdir(parents=True, exist_ok=True)

        # read CSV rows
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            rows = list(reader)

        # build tasks for worker pool AND pre-fill report for skipped items
        tasks = []
        listing_meta = {}  # map from listing_id -> other meta (num_rooms, result_dir)
        report_rows = []   # On commence à remplir le rapport avec les items déjà traités

        print("Checking existing output directories...")
        
        for row in rows:
            listing_id = str(row.get('id','')).strip()
            if not listing_id:
                continue

            images_dir = os.path.join(images_root, listing_id)
            result_dir = str(results_csv_dir / listing_id)

            try:
                num_rooms = int(row.get('num_rooms') or row.get('num_pieces') or 1)
            except Exception:
                num_rooms = 1

            # --- VERIFICATION EXISTENCE ---
            already_done = False
            if os.path.isdir(result_dir):
                # Vérifie s'il y a au moins une image dans le dossier de destination
                # On scanne juste le premier niveau du dossier result_dir
                files_in_dest = os.listdir(result_dir)
                has_image = any(os.path.splitext(f)[1].lower() in check_exts for f in files_in_dest)
                
                if has_image:
                    already_done = True
            
            if already_done:
                # Si déjà traité, on note dans le rapport et on SKIP la tâche
                # print(f"Skipping {listing_id} (already processed)")
                report_rows.append({
                    "listing_id": listing_id,
                    "total": 0,
                    "kept": 0,
                    "selected": 0,
                    "clusters": 0,
                    "selected_paths": "ALREADY_PROCESSED_SKIPPED",
                    "time_s": 0.0
                })
                continue # On passe au listing suivant sans l'ajouter aux tasks
            # ------------------------------

            # Si pas déjà fait, on ajoute à la liste des tâches à traiter
            tasks.append((listing_id, images_dir))
            listing_meta[listing_id] = {"result_dir": result_dir, "num_rooms": num_rooms}

        print(f"Total listings: {len(rows)}. To process: {len(tasks)}. Skipped: {len(report_rows)}.")

        if tasks:
            # use imap_unordered to read images in parallel (I/O bound)
            it = pool.imap_unordered(read_listing_images_for_listing, tasks)

            # iterate as workers complete
            pbar = tqdm(total=len(tasks), desc=f"Listings ({csv_base})")
            for listing_id, images_bytes in it:
                pbar.update(1)
                meta = listing_meta.get(listing_id)
                if meta is None:
                    print(f"[orchestrator] unknown listing returned: {listing_id}")
                    continue

                result_dir = meta["result_dir"]
                num_rooms = meta["num_rooms"]

                # call AI processing in main process (models loaded here)
                try:
                    summary = process_listing_from_bytes(listing_id=listing_id,
                                                         images_bytes=images_bytes,
                                                         results_dir=result_dir,
                                                         clip_cfg=clip_cfg,
                                                         model_vis=model_vis,
                                                         processor_vis=processor_vis,
                                                         device=device,
                                                         batch_size=args.batch_size,
                                                         num_rooms=num_rooms,
                                                         max_pca_components=args.max_pca_components,
                                                         clip_debug=args.clip_debug)
                except Exception as e:
                    print(f"[{listing_id}] pipeline failed: {e}")
                    summary = {"listing_id": listing_id, "total": 0, "kept": 0, "selected": 0, "clusters": 0, "selected_paths": []}

                rel_selected = []
                for p in summary.get("selected_paths", []):
                    try:
                        rel_selected.append(os.path.relpath(p, results_csv_dir))
                    except Exception:
                        rel_selected.append(p)
                
                # Ajout des résultats traités au rapport
                report_rows.append({
                    "listing_id": listing_id,
                    "total": summary.get("total", 0),
                    "kept": summary.get("kept", 0),
                    "selected": summary.get("selected", 0),
                    "clusters": summary.get("clusters", 0),
                    "selected_paths": "|".join(rel_selected),
                    "time_s": summary.get("time_s", 0.0)
                })

            pbar.close()

        # write report.csv for this csv (contains both SKIPPED and PROCESSED)
        report_path = results_csv_dir / "report.csv"
        with open(report_path, "w", newline='', encoding='utf-8') as rf:
            fieldnames = ["listing_id", "total", "kept", "selected", "clusters", "selected_paths", "time_s"]
            writer = csv.DictWriter(rf, fieldnames=fieldnames)
            writer.writeheader()
            for r in report_rows:
                writer.writerow(r)
        print(f"Saved report: {report_path}")

    pool.close()
    pool.join()


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--csv-root', type=str, default='../../output/csv')
    parser.add_argument('-i','--images-root', type=str, default='../../../data/images')
    parser.add_argument('-r','--results-root', type=str, default='../../output/images')
    parser.add_argument('--visual-model', type=str, default='facebook/dinov3-vits16-pretrain-lvd1689m')
    parser.add_argument('--clip-hf-model', type=str, default='openai/clip-vit-base-patch32')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-pca-components', type=int, default=16)
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'])
    parser.add_argument('--num-workers', type=int, default=4, help='workers for CPU I/O (reading images)')
    parser.add_argument('--clip-debug', action='store_true')
    args = parser.parse_args()


    # unify device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== ORCHESTRATOR START ===")
    print(f"device={args.device}, cpu workers={args.num_workers}, batch_size={args.batch_size}")
    process_all(args.csv_root, args.images_root, args.results_root, args)
    print("=== DONE ===")

if __name__ == "__main__":
    main()
