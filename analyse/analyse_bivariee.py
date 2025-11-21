
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
from outils import get_variable_types,clean_outliers,charger_fichier,explode_multilabel_column
from analyse_bivariee_numTonum import showMat_Corr,show_scatter_raw_vs_clean
from analyse_bivariee_cat import boxplot_prix_par_categorie


#--------------------
# PARTIE 1: L'analyse bivariee avec variables numériques
#--------------------

def analyse_numerique_numerique(df):
    print("\n=== Analyse numérique ↔ numérique ===")
    showMat_Corr(df, clean=True)

    variables_scatters = [
        "surface_habitable",
        "nb_pieces",
        "surface_tolale_terrain",
        "nb_chambres",
        "annee_construction",
    ]
    # à modiffier
    for x in variables_scatters:
        if x in df.columns and "prix" in df.columns:
            print(f"\nScatter plot prix vs {x} (brut vs clean)")
            show_scatter_raw_vs_clean(df, x, "prix") # prix par defaut, mais on pourra peut etre comparé à autre chose


#--------------------
# PARTIE 2 : L'analyse bivariee avec varaibles categorielles
#-------------------


def analyse_num_cat(df):
    """
    Analyse bivariée numérique ↔ catégorielle pour le prix.
    """

    print("\n=== Analyse numérique ↔ catégorielle (prix) ===")
    cat_candidates = [
        "type_bien",
        "region",
        "departement",
        "classe_energetique",
        "orientation",
        "anciennete_bien",
    ]
    for col in cat_candidates:
        if col in df.columns:
            print(f"\nBoxplot prix ~ {col}")
            boxplot_prix_par_categorie(df, col, y="prix")




#--------------------
# PARTIE 3: L'analyse bivariee avec varaibles categorielles spéciales (accès exterieur, specificities)
#-------------------

def analyse_multilabel_vs_prix(df, col, sep, top=15):
    """
    Pour une colonne multilabel (ex: 'specificites'),
    calcule et affiche le prix moyen par étiquette.
    """
    if col not in df.columns or "prix" not in df.columns:
        print(f"Colonne {col} ou 'prix' absente du DataFrame.")
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
            moyens[c] = df.loc[mask, "prix"].mean()

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
    plt.show()


def analyse_multilabels(df):
    """
    Analyse des colonnes multilabel 'acces_exterieur' et 'specificites' par rapport au prix.
    """
    print("\n=== Analyse des variables multilabels vs prix ===")

    if "acces_exterieur" in df.columns:
        print("\nEffet des types d'accès extérieur sur le prix (acces_exterieur)")
        analyse_multilabel_vs_prix(df, "acces_exterieur", sep=",")

    if "specificites" in df.columns:
        print("\nEffet des spécificités sur le prix (specificites)")
        analyse_multilabel_vs_prix(df, "specificites", sep="|")




# ======================================
# PARTIE 4 : Analyse géographique
# ======================================

def analyse_geo(df):
    """
    Analyse simple des relations géographiques :
    - Scatter latitude/longitude coloré par prix
    - Prix moyen par département (barplot)
    """
    print("\n=== Analyse géographique ===")

    if {"latitude", "longitude", "prix"}.issubset(df.columns):
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
        plt.show()

    if "departement" in df.columns and "prix" in df.columns:
        prix_dep = df.groupby("departement")["prix"].mean().sort_values(ascending=False)
        plt.figure(figsize=(10, 5))
        prix_dep.plot(kind="bar")
        plt.title("Prix moyen par département")
        plt.ylabel("Prix moyen")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()





def main():
    parser = argparse.ArgumentParser(
    description="Analyse bivariée (corrélations et scatter plots) d’un fichier CSV."
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        required=True,
        help="Chemin vers le fichier CSV à analyser."
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=1,
        help="Nombre de processus à utiliser (optionnel)."
    )

    args = parser.parse_args()  
    df = charger_fichier(args.file)
    #showMat_Corr(df)
    #analyse_num_cat(df)
    #analyse_multilabels(df)
    #analyse_geo(df)





# Point d'entrée
if __name__ == "__main__":
    main()

