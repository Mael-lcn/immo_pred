import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from outils import get_variable_types, charger_fichier



def plot_correlation_circle(pca, components, feature_names):
    """
    Affiche le cercle des corrélations (variables dans l'espace PCA).
    components : pca.components_[0:2] typiquement
    """
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
    plt.show()




def run_pca(df):
    """
    Applique une PCA propre sur les variables numériques filtrées.
    """
    print("\n=== PCA (Analyse en Composantes Principales) ===")

    # Sélectionner les colonnes numériques utiles
    num_cols = get_variable_types(df)[0]
    df_num = df[num_cols].copy()

    # Imputation des valeurs manquantes par la MEDIANE (recommandé pour une PCA)
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

    plot_correlation_circle(pca, pca.components_[0:2], df_num.columns)

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
    plt.show()




def main():
    parser = argparse.ArgumentParser(description="Analyse multivariée (PCA incluse).")
    parser.add_argument("-f", "--file", type=str, required=True)
    args = parser.parse_args()

    df = charger_fichier(args.file)

    run_pca(df)


if __name__ == "__main__":
    main()
