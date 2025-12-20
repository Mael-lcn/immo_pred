
import os
import argparse
import pandas as pd
import glob
import math
import shutil



# Constante pour la reproductibilité
SEED = 42


def load_csvs_from_folder(folder_path):
    """Charge tous les CSV d'un dossier en un seul DataFrame."""
    files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not files:
        return pd.DataFrame()
    
    # On force dtype=str pour éviter les erreurs de parsing mixtes
    df_list = [pd.read_csv(f, sep=";", dtype=str) for f in files]

    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)


def split_and_chunk(args):
    print(f"--- DÉBUT DE LA FRAGMENTATION PROPORTIONNELLE (MAX {args.max_rows} LIGNES/FILE) ---")
    
    # 1. Chargement complet
    print("1. Chargement des données...")

    achat_path = os.path.join(args.input, "achat")
    loc_path = os.path.join(args.input, "location")

    df_achat = load_csvs_from_folder(achat_path)
    df_loc = load_csvs_from_folder(loc_path)

    # --- CORRECTION ICI : Marquage Numérique (0 = Achat, 1 = Location) ---
    if not df_achat.empty: 
        df_achat['dataset_source'] = 0 
    if not df_loc.empty: 
        df_loc['dataset_source'] = 1

    print(f"   [STATS BRUTES] Achat (0): {len(df_achat):,} | Location (1): {len(df_loc):,}".replace(",", " "))

    if df_achat.empty and df_loc.empty:
        print("[ERREUR] Aucun fichier trouvé.")
        return

    # 2. Mélange indépendant (Shuffle)
    # On mélange chaque groupe pour casser l'ordre temporel des fichiers
    if not df_achat.empty:
        df_achat = df_achat.sample(frac=1, random_state=SEED).reset_index(drop=True)
    if not df_loc.empty:
        df_loc = df_loc.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # 3. Calcul des Ratios
    # On arrondit à 5 décimales pour éviter 0.09999999
    val_ratio = round(1.0 - (args.train_ratio + args.test_ratio), 5)
    
    # Sécurité pour éviter -0.0
    if val_ratio < 0: val_ratio = 0.0

    print(f"   -> Ratios : Train={args.train_ratio*100:.1f}% | Test={args.test_ratio*100:.1f}% | Val={val_ratio*100:.1f}%")

    # 4. Découpage Proportionnel (Stratified Split)
    splits = {"train": [], "val": [], "test": []}

    # On boucle sur chaque DF (Achat et Loc) pour appliquer les coupes
    for category_name, df in [("Achat", df_achat), ("Location", df_loc)]:
        if df.empty: continue
    
        n_total = len(df)
        n_train = int(n_total * args.train_ratio)
        n_val = int(n_total * val_ratio)
        # Le reste va dans test (n_total - n_train - n_val)

        # Slicing
        train_part = df.iloc[:n_train]
        val_part = df.iloc[n_train : n_train + n_val]
        test_part = df.iloc[n_train + n_val:]

        # Ajout aux listes globales
        splits["train"].append(train_part)
        splits["val"].append(val_part)
        splits["test"].append(test_part)

        print(f"   -> {category_name:<8} splité : Train={len(train_part):,} | Val={len(val_part):,} | Test={len(test_part):,}")

    # 5. Fusion, Mélange Global et Sauvegarde par Chunks
    print("\n3. Fusion et Sauvegarde fragmentée...")
 
    final_stats = {} # Pour le bilan

    for phase in ["train", "val", "test"]:
        if not splits[phase]:
            continue

        # Fusion (Achat + Location pour cette phase)
        combined_df = pd.concat(splits[phase], ignore_index=True)
        
        if combined_df.empty:
            continue

        # Mélange FINAL (Important pour que le DataLoader ait un mix homogène)
        combined_df = combined_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

        final_stats[phase] = combined_df # Stockage pour bilan

        # Préparation dossier
        phase_dir = os.path.join(args.output, phase)
        if os.path.exists(phase_dir):
            shutil.rmtree(phase_dir) # Nettoyage complet
        os.makedirs(phase_dir, exist_ok=True)

        # --- CHUNKING ---
        total_rows = len(combined_df)
        chunk_size = args.max_rows
        num_files = math.ceil(total_rows / chunk_size)

        print(f"   -> {phase.upper():<5} : {total_rows:,} lignes totales -> {num_files} fichier(s).")

        for i in range(0, total_rows, chunk_size):
            chunk = combined_df.iloc[i : i + chunk_size]
    
            part_num = i // chunk_size
            file_name = f"{phase}_part_{part_num}.csv"
            output_path = os.path.join(phase_dir, file_name)
            
            chunk.to_csv(output_path, sep=";", index=False)

    # --- BILAN FINAL ---
    print("\n" + "="*65)
    print("BILAN DE LA RÉPARTITION (Stratifiée)")
    print("="*65)
    print(f"{'SET':<10} | {'TOTAL':<10} | {'ACHAT (0)':<10} | {'LOC (1)':<10} | {'% ACHAT'}")
    print("-" * 65)
    
    for phase in ["train", "val", "test"]:
        if phase in final_stats and not final_stats[phase].empty:
            df = final_stats[phase]
            
            # On convertit en float puis int pour être sûr (au cas où "0.0")
            sources = pd.to_numeric(df['dataset_source'], errors='coerce').fillna(-1)

            n_a = (sources == 0).sum()
            n_l = (sources == 1).sum()
            total = len(df)
            ratio = n_a / total * 100 if total > 0 else 0
            
            print(f"{phase.upper():<10} | {total:<10,} | {n_a:<10,} | {n_l:<10,} | {ratio:.1f}%")
        else:
            print(f"{phase.upper():<10} | 0          | 0          | 0          | 0.0%")
            
    print("="*65)
    print("Note: Le '% ACHAT' doit être sensiblement le même dans les 3 sets.")
    print(f"Données générées dans : {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Prépare le dataset Deep Learning (Split Proportionnel).")
    parser.add_argument('-i', '--input', type=str, default='../output/processed_csv')
    parser.add_argument('-o', '--output', type=str, default='../output')

    parser.add_argument('--train_ratio', type=float, default=0.75, help="Ratio Train (ex: 0.8).")
    parser.add_argument('--test_ratio', type=float, default=0.01, help="Ratio Test (ex: 0.1). Le reste ira dans Validation.")
    parser.add_argument('--max_rows', type=int, default=1000, help="Lignes max par CSV.")

    args = parser.parse_args()

    # Vérification
    total_ratio = args.train_ratio + args.test_ratio
    if total_ratio > 1.0:
        print(f"[ERREUR] Train + Test > 1.0 ({total_ratio}).")
        return
    
    if args.train_ratio <= 0.0:
        print("[ERREUR] Train ratio doit être > 0.")
        return

    split_and_chunk(args)


if __name__ == '__main__':
    main()

