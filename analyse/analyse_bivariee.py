
import pandas as pd
import seaborn as sns
import argparse
import os
import matplotlib.pyplot as plt
from outils import get_variable_types,clean_outliers,charger_fichier,explode_multilabel_column
from analyse_bivariee_numTonum import showMat_Corr,showScatters_Plots
from analyse_bivariee_cat import boxplot_prix_par_categorie
from outils import load_all_regions


#--------------------
# PARTIE 1: L'analyse bivariee avec variables numériques
#--------------------

def analyse_numerique_numerique(df):

    print("\n=== Analyse numérique ↔ numérique ===")

    # matrice de correlation 
    showMat_Corr(df)

    # scatters plots
    showScatters_Plots(df)



#--------------------
# PARTIE 2 : L'analyse bivariee avec varaibles categorielles
#--------------------

def analyse_num_cat(df):
    """
    Analyse bivariée numérique ↔ catégorielle pour le prix.
    """

    print("\n=== Analyse numérique (prix) ↔ catégorielle  ===")
    cat_candidates = [
        "property_type",
        "property_status"
        "region",
        "department",
        "energy_rating",
        "orientation",
        "region"
        #"anciennete_bien",
    ]
    for col in cat_candidates:
        if col in df.columns:
            print(f"\nBoxplot prix ~ {col}")
            boxplot_prix_par_categorie(df, col, y="price")




#--------------------
# PARTIE 3: L'analyse bivariee avec varaibles categorielles spéciales (accès exterieur, specificities)
#-------------------

def analyse_multilabel_vs_prix(df, col, sep, top=15):
    """
    Pour une colonne multilabel (ex: 'specificites'),
    calcule et affiche le prix moyen par étiquette.
    """
    os.makedirs("plots", exist_ok=True)
    if col not in df.columns or "price" not in df.columns:
        print(f"Colonne {col} ou 'price' absente du DataFrame.")
        return

    dummies = explode_multilabel_column(df, col, sep=sep)
    if dummies.empty:
        print(f"Aucune valeur exploitable pour {col}.")
        return

    # aligner les index
    dummies = dummies.reindex(df.index, fill_value=0)

    # prix moyen par étiquette
    moyens = {}
    for c in dummies.columns:
        mask = dummies[c] == 1
        if mask.sum() > 0:
            moyens[c] = df.loc[mask, "price"].mean()

    if not moyens:
        print(f"Pas de prix moyen calculable pour {col}.")
        return

    s = pd.Series(moyens).sort_values(ascending=False).head(top)

    plt.figure(figsize=(8, 4 + 0.2 * top))
    s.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.xlabel("Prix moyen")
    plt.title(f"Prix moyen en fonction de {col} (Top {top})")
    plt.tight_layout()
    plt.savefig(f"plots/_bivariee_multilabel_price_{col}.png")
    plt.close()

def analyse_multilabels(df):
    """
    Analyse des colonnes multilabel 'acces_exterieur' et 'specificites' par rapport au prix.
    """
    print("\n=== Analyse des variables multilabels vs prix ===")

    if "exterior_access" in df.columns:
        print("\nEffet des types d'accès extérieur sur le prix (acces_exterieur)")
        analyse_multilabel_vs_prix(df, "exterior_access", sep=",")

    if "special_features" in df.columns:
        print("\nEffet des spécificités sur le prix ")
        analyse_multilabel_vs_prix(df, "special_features", sep="|")




def analyse_geo(df):
    os.makedirs("plots", exist_ok=True)

    """
    Analyse simple des relations géographiques :
    - Scatter latitude/longitude coloré par prix
    - Prix moyen par département (barplot)
    """
    print("\n=== Analyse géographique ===")

    if {"latitude", "longitude", "price"}.issubset(df.columns):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df,
            x="longitude",
            y="latitude",
            hue="prix",
            s=10,
            alpha=0.6,
            palette="viridis",
        )
        plt.title("Localisation des biens (couleur = prix)")
        plt.tight_layout()
        plt.savefig("plots/_bivariee_geo_scatter_price.png")
        plt.close()

    if "department" in df.columns and "price" in df.columns:
        prix_dep = df.groupby("department")["price"].mean().sort_values(ascending=False)
        plt.figure(figsize=(10, 5))
        prix_dep.plot(kind="bar")
        plt.title("Prix moyen par département")
        plt.ylabel("Prix moyen")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("plots/_bivariee_geo_price_department.png")
        plt.close()





def main():
    parser = argparse.ArgumentParser(
    description="Analyse bivariée (corrélations et scatter plots) d’un fichier CSV."
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Chemin vers le fichier CSV à analyser."
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=1,
        help="Nombre de processus à utiliser (optionnel)."
    )

    parser.add_argument(
        "-p", "--path", 
        type=str, 
        required=True,
        help="Dossier contenant plusieurs fichiers CSV régionaux"
    )

    args = parser.parse_args()  
    df = load_all_regions(args.path)

    num_cols = get_variable_types(df)[0]
    df= clean_outliers(df, num_cols, 0.01, 0.99)

    analyse_numerique_numerique(df)
    analyse_num_cat(df)
    analyse_multilabels(df)
    #analyse_geo(df)





# Point d'entrée
if __name__ == "__main__":
    main()

