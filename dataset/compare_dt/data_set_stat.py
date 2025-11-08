import os
import argparse
import pandas as pd
import time
import glob



def comput_stat(group):
    """
    Calcule le pourcentage de valeurs manquantes par colonne pour un groupe (property_type).
    Retourne une Series indexée par les noms de colonnes contenant les pourcentages (0-100).
    """
    # pourcentage de valeurs manquantes par colonne
    return group.isna().sum() / len(group) * 100


def run(args):
    t0 = time.monotonic()

    csv_src = args.lbc_csv
    csv_list = glob.glob(os.path.join(csv_src, '*.csv'))

    if not csv_list:
        print(f"[WARN] Aucun fichier csv trouvé dans : {csv_src}")
        return

    dfs = []
    for csv_path in csv_list:
        try:
            dfs.append(pd.read_csv(csv_path, sep=";", quotechar='"'))
        except Exception as e:
            print(f"[WARN] Échec lecture {csv_path} : {e}")

    dt = pd.concat(dfs, ignore_index=True)

    # grouper par property_type et appliquer comput_stat
    # result sera un DataFrame : index = property_type, colonnes = colonnes du dataset, valeurs = % missing
    result = dt.groupby('property_type').apply(comput_stat)

    # Si groupby.apply crée un index multi-niveau (property_type, colonne), on veut remettre en forme
    # mais normalement la forme est déjà (property_type x colonnes). Assurons-nous d'avoir un DataFrame propre.
    if isinstance(result, pd.Series):
        # cas improbable : convertir en DataFrame
        result = result.unstack()

    # result peut avoir un index nommé 'property_type' ; remettons-le explicite
    result.index.name = 'property_type'

    # créer dossier de sortie si besoin
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, 'missing_pct_by_property_type.csv')
    # sauvegarder en CSV avec ; comme séparateur si tu veux garder la cohérence avec tes fichiers sources
    result.to_csv(out_path, sep=';', float_format='%.2f')

    tf = time.monotonic() - t0
    print(f"\nRésultat sauvegardé dans : {out_path}")
    print(f"Shape du résultat : {result.shape}  (lignes=property_type, colonnes=champs du dataset)")
    print("\nAperçu:")
    print(result.head().round(2))
    print(f"\nTemps: {tf:.2f}s.")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lbc', '--lbc_csv', type=str, default='../../input/Lbc/')
    parser.add_argument('-o', '--output', type=str, default='../../output/')
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
