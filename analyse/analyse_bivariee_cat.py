
import seaborn as sns
import matplotlib.pyplot as plt


def boxplot_prix_par_categorie(df, col_cat, y="price"):
    """
    Affiche un boxplot de y en fonction d'une variable catégorielle.
    """
    if col_cat not in df.columns or y not in df.columns:
        print(df.columns)
        print(f"Colonnes {col_cat} ou {y} absentes du DataFrame.")
        return

    nb_modalites = df[col_cat].nunique(dropna=True)
    if nb_modalites > 20:
        print(f"{col_cat} a trop de modalités ({nb_modalites}), boxplot ignoré.")
        return

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x=col_cat, y=y)
    plt.title(f"{y} en fonction de {col_cat}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
