# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from outils import load_all_regions
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.cluster import KMeans 
import os
from outils import get_variable_types, charger_fichier


def plot_correlation_circle(pca, components, feature_names,axe_x=0, axe_y=1):
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 8))
    
    # Cercle unité
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
    plt.gca().add_artist(circle)
    
    # Axes
    plt.axhline(0, color='grey', linewidth=1)
    plt.axvline(0, color='grey', linewidth=1)

    # Vecteurs
    for i, (x, y) in enumerate(zip(components[0], components[1])):
        plt.arrow(0, 0, x, y, 
                  head_width=0.03, head_length=0.03, 
                  linewidth=1, color='blue')
        plt.text(x * 1.1, y * 1.1, feature_names[i], fontsize=10)

    plt.xlabel("Axe 1")
    plt.ylabel("Axe 2")
    plt.title("Cercle des corrélations (PCA)")
    plt.grid(True)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.tight_layout()
    plt.savefig(f"plots/_pca_correlation_circle_axes_{axe_x}_{axe_y}.png")
    plt.close()




def run_pca(df):
    """
    Applique une PCA propre sur les variables numériques filtrées.
    """
    print("\n=== PCA (Analyse en Composantes Principales) ===")

    # Sélectionner les colonnes numériques utiles
    num_cols = get_variable_types(df)[0]
    df_num = df[num_cols].copy()

    df_num = df_num.fillna(df_num.median(numeric_only=True))


    if df_num.shape[1] < 2:
        print("Pas assez de variables numériques pour une PCA.")
        return

    print(f"Variables utilisées dans la PCA : {list(df_num.columns)}")

    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num)

    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    plot_correlation_circle(pca, pca.components_[0:2], df_num.columns,1,2) # Axes 1 vs 2
    plot_correlation_circle(pca, pca.components_[[0, 2]], df_num.columns,1,3) # Axes 1 vs 3
    plot_correlation_circle(pca, pca.components_[1:3], df_num.columns,2,3) # Axes 2 vs 3
    plot_correlation_circle(pca, pca.components_[[0, 3]], df_num.columns,1,4) # Axes 1 4
    # pas besoin d'observer les autres axes


    print( "TESSSSSSSSST", pca.components_[0])
    # Variance expliquée
    explained = pca.explained_variance_ratio_
    print("\nVariance expliquée par axe :")
    for i, v in enumerate(explained):
        print(f"Axe {i+1} : {v*100:.2f}%")

    # 5. Scree plot (graphique des eigenvalues)
    plt.figure(figsize=(8,4))
    plt.plot(np.arange(1, len(explained)+1), explained, marker='o')
    plt.title("Scree Plot (Variance expliquée par composante)")
    plt.xlabel("Composante principale")
    plt.ylabel("Variance expliquée")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/_pca_scree_plot.png")
    plt.close()

def k_means(df):
    k=4
    # 1) Récupérer les noms de colonnes numériques
    num_cols = get_variable_types(df)[0]   # <- Index des colonnes numériques

    X = df[num_cols].dropna()  

    # 3) Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4) K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)

    # 5) Remettre les labels dans le df en respectant l’index
    df.loc[X.index, "cluster"] = labels

    # 6) Résumé par cluster
    cluster_summary = df.groupby("cluster")[num_cols].mean().round(2)
    print(f"=== Cluster Summary (k={k}) ===")
    print(cluster_summary)

    return df, cluster_summary
    



def find_number_clusters(df):
    os.makedirs("plots", exist_ok=True)
    num_cols = get_variable_types(df)[0]
        # 2) Données (et plus les noms de colonnes !)
    X = df[num_cols].dropna()

 #on doit standardiser
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4) Méthode du coude
    inertias = []
    K_range = range(2, 10)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8,4))
    plt.plot(K_range, inertias, marker='o')
    plt.xlabel("Nombre de clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Méthode du coude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/_kmeans_elbow_method.png")

    # 5) Scores silhouette
    print("Scores silhouette :")
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"k={k}, silhouette={score:.3f}")







def main():
    parser = argparse.ArgumentParser(description="Analyse multivariée (PCA incluse).")
    parser.add_argument(
        "-p", "--path", 
        type=str, 
        required=True,
        help="Dossier contenant plusieurs fichiers CSV régionaux"
    )
    args = parser.parse_args()
    df = load_all_regions(args.path)



    mapping_etat = {
        "Travaux à prévoir": 0,
        "À rafraichir": 1,
        "Bon état": 2,
        "Rénové": 3,
        "Très bon état": 4
    }
    df["etat_bien_num"] = df["property_status"].map(mapping_etat)


    run_pca(df)
    df = pd.get_dummies(df, columns=["region"], drop_first=True)
    df = pd.get_dummies(df, columns=["property_type"], prefix="type")
    #find_number_clusters(df)
    #k_means(df)
  


if __name__ == "__main__":
    main()
