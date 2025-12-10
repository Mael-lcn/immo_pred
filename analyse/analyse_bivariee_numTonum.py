
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from outils import get_variable_types,clean_outliers
import os 
# m atrice de corrélation 
def showMat_Corr(df,low=0.01, high=0.99):
    os.makedirs("plots", exist_ok=True)
    #petit retour: le prix est mal corrélé à la surface habitable même après avoir clean les outliers (bizarre)
    # On a vu dans l'analyse univariee que d'autres types de biens que Maison/appartemnt étaient présent, ça doit jouer 

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
    plt.savefig(f"plots/univ_cat_correlation_matrix.png")
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
            show_scatter_raw_vs_clean(df, x, "price") # prix par defaut, mais on pourra peut etre comparé à autre chose





            

def show_scatter_raw_vs_clean(df, x, y="price", low=0.01, high=0.99):
    """
    Affiche deux scatter plots côte à côte :
    - à gauche : données brutes
    - à droite : données nettoyées par quantiles (low, high)
    """

    # 1) Sous-dataframe avec uniquement les deux colonnes
    df_xy = df[[x, y]].dropna()

    if df_xy.empty:
        print(f"Aucune donnée disponible pour {x} et {y}.")
        return

    # 2) Version nettoyée pour la visualisation
    df_clean = clean_outliers(df_xy, [x, y], low=low, high=high)

    # 3) Création des deux graphiques côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # --- Scatter brut ---
    axes[0].scatter(df_xy[x], df_xy[y], alpha=0.3)
    axes[0].set_title("Données brutes")
    axes[0].set_xlabel(x)
    axes[0].set_ylabel(y)
    axes[0].grid(True)

    # --- Scatter clean ---
    axes[1].scatter(df_clean[x], df_clean[y], alpha=0.3)
    axes[1].set_title(f"Données nettoyées ({int(low*100)}–{int(high*100)}ᵉ quantile)")
    axes[1].set_xlabel(x)
    axes[1].set_ylabel(y)
    axes[1].grid(True)

    fig.suptitle(f"{y} en fonction de {x} : brut vs clean", y=1.02)
    plt.tight_layout()
    plt.savefig(f"plots/scatter_{x}_vs_{y}.png")
    plt.close()