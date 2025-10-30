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
        "initial_rows": len(df),
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

    # Colonnes vides
    empty_cols_nan = df.columns[df.isna().all()].tolist()
    empty_cols_str = df.columns[(df == '').all()].tolist()
    empty_cols = list(set(empty_cols_nan + empty_cols_str))
    df = df.drop(columns=empty_cols)
    stats["empty_removed"] = len(empty_cols)

    stats["final_rows"] = len(df)

    return df, stats


def worker(list_csv, output_dir):
    """
    Traite un lot de fichiers CSV et retourne les stats.
    """
    if not list_csv:
        return None

    try:
        dfs = [pd.read_csv(csv_path, sep=";", quotechar='"') for csv_path in list_csv]

        df_concat = pd.concat(dfs, ignore_index=True, sort=False)

        df_final, stats = filtre(df_concat)

        # Écriture du fichier de sortie
        output_file = os.path.join(output_dir, os.path.basename(list_csv[-1]))
        df_final.to_csv(output_file, sep=";", index=False, quoting=csv.QUOTE_MINIMAL)

    except Exception as e:
        error_file = os.path.join(output_dir, "erreur.txt")
        with open(error_file, "a", encoding="utf-8") as f:
            f.write(f"[ERROR] Lot {list_csv[:5]}... : {e}\n")

    return stats


def run(args):
    t0 = time.monotonic()
    os.makedirs(args.output, exist_ok=True)

    csv_list = glob.glob(os.path.join(args.input, '*.csv'))

    if not csv_list:
        print(f"[WARN] Aucun fichier CSV trouvé dans : {args.input}")
        return

    # Découpage en lots
    chunksize = math.ceil(len(csv_list) / max(1, args.workers))
    data = [csv_list[i:i + chunksize] for i in range(0, len(csv_list), chunksize)]
    num_processes = min(args.workers, len(data))

    print(f" Traitement {len(csv_list)} fichiers -> {len(data)} lots, processes={num_processes}")

    fun = partial(worker, output_dir=args.output)

    stats_global = {
        "initial_rows": 0,
        "final_rows": 0,
        "na_removed": 0,
        "dup_removed": 0,
        "empty_removed": 0
    }

    with multiprocessing.Pool(processes=num_processes) as pool:
        for batch_stats in tqdm(pool.imap_unordered(fun, data), total=len(data)):
            for key in stats_global:
                stats_global[key] += batch_stats[key]

    dt = time.monotonic() - t0
    print(f"\nTemps total: {dt:.2f}s. CSV concaténés écrits dans : {args.output}")

    erreur_file = os.path.join(args.output, "erreur.txt")
    if os.path.exists(erreur_file):
        print(f"[WARN] Certaines erreurs ont été enregistrées dans : {erreur_file}")

    # Résumé final
    print("\n=== Résumé global du nettoyage ===")
    print(f"Lignes initiales: {stats_global['initial_rows']}")
    print(f"Lignes finales  : {stats_global['final_rows']}")
    print(f"Lignes supprimées (NaN)   : {stats_global['na_removed']}")
    print(f"Lignes supprimées (doublons) : {stats_global['dup_removed']}")
    print(f"Colonnes vides supprimées     : {stats_global['empty_removed']}")
    print(f"Pourcentage de données gardée : {stats_global['final_rows'] / stats_global['initial_rows'] * 100}%")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='input/LBC')
    parser.add_argument('-o', '--output', type=str, default='output/csv')
    parser.add_argument('-w', '--workers', type=int, default=multiprocessing.cpu_count()-1)
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
