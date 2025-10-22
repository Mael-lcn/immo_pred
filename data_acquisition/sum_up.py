import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import glob


def run(args):
    csv_folder = args.csv
    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)

    # Liste tous les fichiers CSV
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    print(f"{len(csv_files)} fichiers CSV trouvés dans {csv_folder}")

    # Lire et fusionner tous les CSV
    df_list = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f" Erreur en lisant {f} : {e}")

    if not df_list:
        print(" Aucun CSV valide à fusionner.")
        return

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"\nDataFrame fusionné : {len(combined_df)} lignes, {len(combined_df.columns)} colonnes")

    # Sélection des colonnes numériques
    numeric_cols = combined_df.select_dtypes(include='number').columns
    if not numeric_cols.any():
        print(" Aucun champ numérique trouvé pour calculer les moyennes et distributions.")
        return

    # Calcul des moyennes et stats de base
    numeric_summary = combined_df[numeric_cols].agg(['mean', 'min', 'max', 'std'])
    summary_csv_path = os.path.join(output_folder, 'numeric_summary.csv')
    numeric_summary.to_csv(summary_csv_path)
    print(f"\n Statistiques numériques sauvegardées : {summary_csv_path}")

    # Génération des distributions (histogrammes) pour chaque variable numérique
    dist_folder = os.path.join(output_folder, 'distributions')
    os.makedirs(dist_folder, exist_ok=True)

    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        combined_df[col].hist(bins=30)
        plt.title(f"Distribution de {col}")
        plt.xlabel(col)
        plt.ylabel("Fréquence")
        plt.tight_layout()
        plot_path = os.path.join(dist_folder, f"{col}_distribution.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Distribution sauvegardée : {plot_path}")

    print("\nToutes les statistiques et distributions ont été générées.")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv', type=str, default='../../data/SeLoger/achat/csv')
    parser.add_argument('-o', '--output', type=str, default='../output/')
    args = parser.parse_args()

    run(args)



if __name__ == '__main__':
    main()
