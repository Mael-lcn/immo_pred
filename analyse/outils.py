import sys
import pandas as pd
import glob
import os

def get_variable_types(dataframe: pd.DataFrame):
    """Retourne les listes de colonnes quantitatives et qualitatives filtrés ."""

    num_cols = dataframe.select_dtypes(include="number").columns
    cols_geo = ["latitude", "longitude","postal_code"]     # on enlève les colonnes latitude et longitude car ce sont des géographiques, il faut les traiter à part
    cols_id  = ["id","type_quartier_id","id_quartier","dataset_source"]             # identifiants à exclure (on peut pas faire d'études dessus)
    cols_typeVente =["type_vente"] # toujours à 0 car on a que des pros ici
    cols_nums_to_exclude = cols_geo + cols_id +cols_typeVente
    #Colonnes filtrées
    num_cols = [ c for c in dataframe.select_dtypes(include="number").columns if c not in cols_nums_to_exclude]
    # les varaibles categorielles
    cols_cat_to_ignore = [                                      
        "reference",
        "description",
        "images",
        "titre",
        "date_publication"
    ]
    """
    Raisons : 
    id, reference,images_urls => 1 valeur equivalent à 1 annonce
    description => //
    titre => //

    """
    cat_cols = [c for c in dataframe.select_dtypes(exclude="number").columns if c not in cols_cat_to_ignore]
    return num_cols, cat_cols



def clean_outliers(df, cols, low=0.01, high=0.99):
    df_clean = df.copy()
    for c in cols:
        q_low = df[c].quantile(low)
        q_high = df[c].quantile(high)
        df_clean[c] = df[c].clip(q_low, q_high)
    return df_clean





def explode_multilabel_column(df, col, sep=','):
    """
    Transforme une colonne multivaluée en colonnes binaires (one-hot).
    Ex: 'terrace,garden' -> terrace=1, garden=1
    """
    # On travaille sur une copie pour éviter les effets de bord
    s = df[col].fillna('')

    # On split + explode tout en gardant l'index d'origine
    exploded = s.str.split(sep).explode().str.strip()

    # On enlève les valeurs vides
    exploded = exploded[exploded != '']

    if exploded.empty:
        # Rien à encoder
        return pd.DataFrame(index=df.index)

    # One-hot encoding
    dummies = pd.get_dummies(exploded, prefix=col)

    # Regrouper par index d'origine (level=0)
    dummies = dummies.groupby(level=0).sum()

    # S'assurer qu'on a bien un index aligné avec le df d'origine
    dummies = dummies.reindex(df.index, fill_value=0)

    return dummies

def charger_fichier(fichier_csv):
    """
    Charge le fichier CSV spécifié par l'utilisateur via les arguments.
    """
    print(f"\n Chargement du fichier : {fichier_csv}")
    try:
        df = pd.read_csv(fichier_csv, sep=";", engine="python")
        print("Fichier chargé avec succès !")
        return df
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        sys.exit(1)






# ou from .outils import charger_fichier, selon où est la fonction

def load_all_regions(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        raise ValueError("Aucun fichier CSV trouvé dans ce dossier.")

    print(f"{len(csv_files)} fichiers trouvés :")
    for f in csv_files:
        print(" -", os.path.basename(f))

    df_list = []
    for file in csv_files:
        try:
            df_temp = charger_fichier(file)

            # ❌ IMPORTANT : ne PAS écraser la colonne "region"
            # ❌ Ne plus faire : df_temp["region"] = os.path.basename(file)

            df_list.append(df_temp)

        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture de {file} : {e}")
            continue

    df = pd.concat(df_list, ignore_index=True)
    print(f"\nDataset fusionné : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df







