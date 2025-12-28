
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from outils import get_variable_types,clean_outliers
import os 
# m atrice de corrélation 
def showMat_Corr(df,low=0.01, high=0.99):
    os.makedirs("plots", exist_ok=True)
   

    # récupérer les colonnes numériques filtrées
    num_cols = get_variable_types(df)[0]

    # ne surtout pas faire dropna() ici
    df_num = df[num_cols]

    df_num = clean_outliers(df_num, num_cols,low,high)
    # enlever les colonnes constantes (facultatif mais propre)
    df_num = df_num.loc[:, df_num.nunique() > 1]

    corr = df_num.corr(method="pearson")
    # vérifier si la matrice est pleine de NaN
    if corr.isna().all().all():
        print("Corrélation impossible : trop de NaN ou colonnes constantes.")
        print("Colonnes numériques :", df_num.columns.tolist())
        return

    # tracé de la heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=len(df_num.columns) <= 15,   # annot si lisible
        fmt=".2f",
        cmap="coolwarm",
        cbar=True
    )
    plt.title("Matrice de corrélation")
    plt.tight_layout()
    plt.savefig(f"plots/correlation_matrix.png")
    plt.close()

def showScatters_Plots(df):
    os.makedirs("plots", exist_ok=True)
    variables_scatters = [
        "living_area_sqm",
        "total_land_area_sqm",
        "num_rooms",
        "num_bedrooms",
        "num_bathrooms",
        "num_parking_spaces",
        "year_built",
    ]
    # à modiffier
    for x in variables_scatters:
        if x in df.columns and "price" in df.columns:
            print(f"\nScatter plot prix vs {x} (brut vs clean)")
            show_scatter(df, x, "price") # prix par defaut, mais on pourra peut etre comparé à autre chose





            


def show_scatter(df, x, y="price"):
    # Sélection des colonnes et suppression des NaNs
    df_xy = df[[x, y]].dropna()

    if df_xy.empty:
        print(f"Aucune donnée disponible pour {x} et {y}.")
        return

    # Création d'un seul graphique (plus besoin de subplots)
    plt.figure(figsize=(8, 6))

    # Affichage du scatter plot
    plt.scatter(df_xy[x], df_xy[y], alpha=0.3)
    
    # Titre et labels
    plt.title(f"{y} en fonction de {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)

    # Sauvegarde
    plt.tight_layout()
    plt.savefig(f"plots/scatter_{x}_vs_{y}.png")
    plt.close()