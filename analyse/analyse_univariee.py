# 1. Importation de pandas
import pandas as pd
import matplotlib.pyplot as plt
from outils import get_variable_types,explode_multilabel_column,charger_fichier,clean_outliers
import argparse
import math
from outils import load_all_regions
import numpy as np
import seaborn as sns
import os
pd.set_option('display.max_columns', None) # Affiche toutes les colonnes
pd.set_option('display.width', 1000)       # Évite les retours à la ligne moches
# appliquer des derniers filtres  plus tard (viager, type Appartement, et explique pourquoi on a besoin de les enlever) => crée des anomalies statistiques meme en retirant les outliers trop important
# trouver un moyen sympa d'analyser les colonnes geographiques (lat,long), une plus-value à region,ville

def affiche_caracteristique_par_defaut(df):
    #Objectif: décrire simplement le jet de donnée
    print("\nColonnes présentes dans le fichier :")
    print(df.columns.tolist())

    # Colonnes quantitatives (par defaut)
    colonnes_quanti = df.select_dtypes(include=["number"]).columns.tolist()

    # Colonnes qualitatives (par defaut)
    colonnes_quali = df.select_dtypes(exclude=["number"]).columns.tolist()

    
    print(" Variables quantitatives :")
    print(colonnes_quanti)

    print("\n🔤 Variables qualitatives :")
    print(colonnes_quali)

    print("Aperçu du jeu de données :")
    print(df.head())

    print("\nInformations générales :")
    print(df.info())
 
    # 6. Statistiques descriptives pour les variables numériques
    print("\nStatistiques descriptives :")
    print(df.describe())


def stats_univariees(df,low=0.01,high=0.99): # Les stats varient en fonction de l'élimination des outliers, les paramètres sont donc personnalisables

    #les variables numériques 

    num_cols,cat_cols = get_variable_types(df)
    df_clean = clean_outliers(df, num_cols, low=low, high=high)
    print("*******")
    print(df_clean["price"].describe())
    print("*******")

    desc_num = df_clean[num_cols].describe().T
    desc_num["skew"] = df_clean[num_cols].skew() # on regarde l'asymétrie
    desc_num["kurtosis"] = df_clean[num_cols].kurt() # ici l'aplatissement
    print("\n=== Statistiques descriptives (numériques) ===")
    print(desc_num)

    
    print("\n=== Statistiques descriptives (catégorielles) ===")
    for col in cat_cols:
        print(f"\nVariable : {col}")
        print(df[col].value_counts(dropna=False))
        # pour le comptage des variables  acces_exterieur et specifities, ce serait peut être intéressant de faire un plot du nombre de d'acces exterieurs et de specifities, avec la moyenne ,ecarts types , outliers, ....
        # faudra créer un bloc spécifique pour analyser ces 2 variables



def detect_outliers_iqr(df, q1=0.01,q3=0.99):
    # petit retour : Les prix ont enormément d'outliers  et c'est normal, il faudra peut être mettre des quantiles spécifiques à eux (0.5,0.95 par exemple) ou log + detect outliers 
    # il y a un outlier surprenant, sur l'année de construction, ce sont des professionels qui ont enregistré l'annonce, on s'attenderait pas à voir des -2300 ou 4000.
    num_cols= get_variable_types(df)[0]
    outlier_summary = {}

    for col in num_cols:
        Q1 = df[col].quantile(q1)
        Q3 = df[col].quantile(q3)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_summary[col] = mask.sum()

    print(f"\n=== Outliers (méthode IQR), quantiles pris : ({q1},{q3})===")
    for col, nb in outlier_summary.items():
        print(f"{col} : {nb} outliers")

    

def check_data_quality(df):
    print("\n=== Valeurs manquantes par colonne ===")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing)

    print("\n=== Doublons ===")
    print(df.duplicated().sum())
    print("\n================")


def showBoxPlots(df, low=0.01, high=0.99):
    os.makedirs("plots", exist_ok=True)
    num_cols = get_variable_types(df)[0]

    # Nettoyage des outliers
    df_clean = clean_outliers(df, num_cols, low, high)

    for col in num_cols:
        plt.figure(figsize=(6, 4))

        sns.violinplot(
            data=df_clean,
            y=col,                  # variable numérique en Y
            x=None,                 # pas de variable catégorielle
            inner="box",
            cut=0,
            density_norm="width"
        )

        plt.title(f"Violin plot de {col}")
        plt.ylabel(col)
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"plots/univ_violin_{col}.png")
    
    plt.close()


#montre les histogrammes des varaibles numériques
def showHistoPlots(df,low=0.01,high=0.99):
    os.makedirs("plots", exist_ok=True)
    num_cols= get_variable_types(df)[0]
    df_clean = clean_outliers(df, num_cols,low, high)


    for col in num_cols:
        plt.figure(figsize=(6,4))
        plt.hist(df_clean[col].dropna(), bins='auto')
        plt.title(f"Histogramme de {col}")
        plt.xlabel(col)
        plt.ylabel("Fréquence")
        plt.grid(True)
        plt.savefig(f"plots/univ_hist_{col}.png")
        plt.close()

def showCatPlots(df, max_categories=20): # je limite le nombre de categorie pour une variable car si y'en a trop c'est plus representable par un plot 
    os.makedirs("plots", exist_ok=True)
    # Variables catégorielles

    cat_cols = get_variable_types(df)[1]

    # Colonnes à ignorer (trop de modalités)
    filtres_supplementaires = [
        "special_features",
        "outside",
        "city",
    ]
    
    # On retire les colonnes ignorées
    cat_cols = [c for c in cat_cols if c not in filtres_supplementaires]

    if len(cat_cols) == 0:
        print("Aucune colonne catégorielle à afficher.")
        return

    for col in cat_cols:
        plt.figure(figsize=(8, 4))

        vc = df[col].value_counts(dropna=False)

        # Si trop de catégories, on montre juste le top
        if vc.shape[0] > max_categories:
            vc = vc.head(max_categories)

        vc.plot(kind="bar")

        plt.title(col)
        plt.ylabel("Effectif")
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(f"plots/univ_cat_{col}.png")
        plt.close()

def plot_multilabel_counts(df, col, sep, top=20):
    os.makedirs("plots", exist_ok=True)
    d = explode_multilabel_column(df, col, sep)
    counts = d.sum().sort_values(ascending=False).head(top)

    plt.figure(figsize=(8, 4 + 0.2*top))
    counts.plot(kind='barh')
    plt.title(f"Top {top} valeurs de {col}")
    plt.xlabel("Nombre d'annonces")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"plots/univ_multilabel_{col}.png")
    plt.close()

def afficher_repartition_biens(df):
    print("\n=== Répartition Appartements vs Maisons ===")
    
    # Compte les valeurs (ex: 22000 Maisons, 18000 Apparts)
    comptes = df["property_type"].value_counts()
    print(comptes)
    
    print("\n--- En Pourcentage ---")
    # Affiche en % (ex: 0.55 pour 55%)
    pourcentages = df["property_type"].value_counts(normalize=True) * 100
    print(pourcentages)


def main():
    parser = argparse.ArgumentParser(
        description="Analyse univariée d’un fichier CSV."
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

    print(df["price"].describe())
    detect_outliers_iqr(df)
    #stats_univariees(df)
    
 #   affiche_caracteristique_par_defaut(df)
  #  check_data_quality(df)
    #showCatPlots(df)
    #showBoxPlots(df)   
    #howHistoPlots(df)
    #afficher_repartition_biens(df)


     




# Point d'entrée
if __name__ == "__main__":
    main()

