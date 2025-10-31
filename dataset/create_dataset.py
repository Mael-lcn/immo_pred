import os
import argparse
import pandas as pd
import time
import glob
from tqdm import tqdm
from functools import partial
import multiprocessing
import math
import csv
from collections import Counter



def filtre(df):
    """
    Nettoie et transforme la DataFrame concaténée :
      - Supprime les lignes contenant des valeurs NaN
      - Supprime les doublons
      - Supprime les colonnes vides (toutes valeurs NaN ou '')
      - Réinitialise l'index

    Retourne un tuple : (DataFrame nettoyée, stats)
    """
    stats = {
        "initial_rows": len(df) if df is not None else 0,
        "final_rows": 0,
        "na_removed": 0,
        "dup_removed": 0,
        "empty_removed": 0
    }

    if df is None or df.empty:
        return df, stats

    # Lignes avec NaN
    df = df.dropna()
    stats["na_removed"] = stats["initial_rows"] - len(df)

    # Doublons
    df = df.drop_duplicates(ignore_index=True)
    stats["dup_removed"] = (stats["initial_rows"] - stats["na_removed"]) - len(df)

    # Colonnes vides (toutes NaN ou toutes chaînes vides)
    empty_cols_nan = df.columns[df.isna().all()].tolist()
    empty_cols_str = df.columns[(df == '').all()].tolist()
    empty_cols = list(set(empty_cols_nan + empty_cols_str))
    df = df.drop(columns=empty_cols)
    stats["empty_removed"] = len(empty_cols)

    stats["final_rows"] = len(df)

    return df, stats


def worker(list_csv, output_dir):
    """
    Traite un lot de fichiers CSV :
      - concatène et filtre (comme avant)
      - écrit le CSV de sortie pour le lot
      - retourne (stats, counter) où counter est un collections.Counter des seller_name du lot (après filtre)
    """
    # Valeurs par défaut en cas d'erreur
    default_stats = {
        "initial_rows": 0,
        "final_rows": 0,
        "na_removed": 0,
        "dup_removed": 0,
        "empty_removed": 0
    }
    counter = Counter()

    if not list_csv:
        return default_stats, counter

    try:
        # Lire tous les CSV du lot
        dfs = []
        for csv_path in list_csv:
            # Lecture complète (on pourrait optimiser en lecture partielle si nécessaire)
            dfs.append(pd.read_csv(csv_path, sep=";", quotechar='"', dtype=str, low_memory=False))

        df_concat = pd.concat(dfs, ignore_index=True, sort=False)

        # Appliquer le filtre
        df_final, stats = filtre(df_concat)

        # Écriture du fichier de sortie (nom basé sur le dernier fichier du lot)
        output_file = os.path.join(output_dir, os.path.basename(list_csv[-1]))
        try:
            df_final.to_csv(output_file, sep=";", index=False, quoting=csv.QUOTE_MINIMAL)
        except Exception as e:
            # Si écriture échoue, on log l'erreur mais on continue
            error_file = os.path.join(output_dir, "erreur.txt")
            with open(error_file, "a", encoding="utf-8") as f:
                f.write(f"[ERROR WRITE] Lot {list_csv[:5]}... : {e}\n")

        # Construire le counter à partir de df_final (après filtre)
        if "seller_name" in df_final.columns:
            sellers = (
                df_final["seller_name"]
                .dropna()
                .astype(str)
                .map(lambda s: s.strip())
            )
            sellers = sellers[sellers != ""]
            counter.update(sellers.tolist())

    except Exception as e:
        # Log erreur et renvoyer defaults
        error_file = os.path.join(output_dir, "erreur.txt")
        with open(error_file, "a", encoding="utf-8") as f:
            f.write(f"[ERROR] Lot {list_csv[:5]}... : {e}\n")
        stats = default_stats
        counter = Counter()

    return stats, counter


def run(args):
    t0 = time.monotonic()
    os.makedirs(args.output, exist_ok=True)

    csv_list = glob.glob(os.path.join(args.input, '*.csv'))

    if not csv_list:
        print(f"[WARN] Aucun fichier CSV trouvé dans : {args.input}")
        return

    # Découpage en lots (équilibré selon le nombre de workers)
    chunksize = math.ceil(len(csv_list) / max(1, args.workers))
    data = [csv_list[i:i + chunksize] for i in range(0, len(csv_list), chunksize)]
    num_processes = min(args.workers, len(data))

    print(f"Traitement {len(csv_list)} fichiers -> {len(data)} lots, processes={num_processes}")

    fun = partial(worker, output_dir=args.output)

    stats_global = {
        "initial_rows": 0,
        "final_rows": 0,
        "na_removed": 0,
        "dup_removed": 0,
        "empty_removed": 0
    }

    sellers_counter = Counter()

    # Pool multiprocessing
    with multiprocessing.Pool(processes=num_processes) as pool:
        # imap_unordered retourne des (stats, counter)
        for result in tqdm(pool.imap_unordered(fun, data), total=len(data)):
            if result is None:
                continue
            batch_stats, batch_counter = result

            # Sécurité : utiliser .get pour éviter d'échouer si clé manquante
            for key in stats_global:
                stats_global[key] += batch_stats.get(key, 0)

            sellers_counter.update(batch_counter)

    dt = time.monotonic() - t0
    print(f"\nTemps total: {dt:.2f}s. CSV concaténés écrits dans : {args.output}")

    # Sauvegarder les résultats des counts
    counts_csv = os.path.join(args.output, "seller_counts.csv")
    df_counts = pd.DataFrame(sellers_counter.items(), columns=["seller_name", "count"])
    df_counts = df_counts.sort_values("count", ascending=False).reset_index(drop=True)
    try:
        df_counts.to_csv(counts_csv, sep=";", index=False, quoting=csv.QUOTE_MINIMAL)
    except Exception as e:
        print(f"[WARN] Impossible d'écrire {counts_csv} : {e}")


    # Fichier texte des sellers uniques (triés)
    sellers_txt = os.path.join(args.output, "unique_sellers.txt")
    try:
        with open(sellers_txt, "w", encoding="utf-8") as f:
            for s in sorted(sellers_counter.keys()):
                f.write(s + "\n")
    except Exception as e:
        print(f"[WARN] Impossible d'écrire {sellers_txt} : {e}")

    # Résumé & top
    print(f"\nNombre total de sellers uniques: {len(sellers_counter)}")
    print("Top 10 sellers (seller -> count) :")
    for seller, cnt in sellers_counter.most_common(10):
        print(f"  {seller} -> {cnt}")

    erreur_file = os.path.join(args.output, "erreur.txt")
    if os.path.exists(erreur_file):
        print(f"[WARN] Certaines erreurs ont été enregistrées dans : {erreur_file}")

    # Résumé final (sécuriser la division)
    print("\n=== Résumé global du nettoyage ===")
    print(f"Lignes initiales: {stats_global['initial_rows']}")
    print(f"Lignes finales  : {stats_global['final_rows']}")
    print(f"Lignes supprimées (NaN)   : {stats_global['na_removed']}")
    print(f"Lignes supprimées (doublons) : {stats_global['dup_removed']}")
    print(f"Colonnes vides supprimées     : {stats_global['empty_removed']}")
    kept_pct = (stats_global['final_rows'] / stats_global['initial_rows'] * 100) if stats_global['initial_rows'] else 0
    print(f"Pourcentage de données gardée : {kept_pct:.2f}%")

    print(f"\nFichiers générés :\n - {counts_csv}\n - {sellers_txt}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='input/LBC')
    parser.add_argument('-o', '--output', type=str, default='output/csv')
    parser.add_argument('-w', '--workers', type=int, default=max(1, multiprocessing.cpu_count()-1))
    args = parser.parse_args()

    run(args)

if __name__ == '__main__':
    main()
