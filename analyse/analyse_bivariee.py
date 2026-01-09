
import pandas as pd
import seaborn as sns
import geopandas as gpd
import argparse
import os
import matplotlib.pyplot as plt
from outils import get_variable_types,clean_outliers,charger_fichier,explode_multilabel_column
from analyse_bivariee_numTonum import showMat_Corr,showScatters_Plots
from analyse_bivariee_cat import boxplot_prix_par_categorie
from scipy.stats import f_oneway
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


def analyse_anova(df):

    print("\n=== Test ANOVA (Variables catégorielles vs Prix) ===")
    
    # Variables à tester
    target_col = "price"
    candidates = ["energy_rating", "orientation", "region"]
    
    if target_col not in df.columns:
        print(f"Erreur: La colonne cible '{target_col}' est absente.")
        return

    for col in candidates:
        if col not in df.columns:
            print(f"\n[!] Colonne manquante : {col}")
            continue

        #on retire les NaN pour la paire (col, price)
        data_clean = df[[col, target_col]].dropna()

        # (liste de listes des prix par catégorie)
        groups = [group[target_col].values for name, group in data_clean.groupby(col)]

        # Il faut au moins 2 groupes pour comparer
        if len(groups) < 2:
            print(f"\n--- ANOVA : {col} ---")
            print("Pas assez de catégories pour effectuer le test.")
            continue

        # 3. Calcul de la F-statistic et P-value
        stat, p_value = f_oneway(*groups)

        # 4. Affichage des résultats
        print(f"\n--- ANOVA : Prix ~ {col} ---")
        print(f"F-statistic : {stat:.2f}")
        # Affichage scientifique pour les p-values très petites
        print(f"P-value     : {p_value:.4e}") 
        
        # Interprétation
        alpha = 0.05
        if p_value < alpha:
            print(" Résultat SIGNIFICATIF : Les prix moyens varient selon cette catégorie.")
        else:
            print(" Résultat NON SIGNIFICATIF : Pas de preuve de différence de prix.")


import unicodedata

def normalize_name(name):
    """Nettoie les noms pour maximiser les chances de correspondance"""
    if pd.isna(name): return ""
    # Enlève les accents (ex: Île -> Ile)
    n = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    # Tout en minuscule, enlève les tirets et les apostrophes
    return n.lower().replace("-", " ").replace("'", " ").strip()
def map_prix_par_region(df, col_region="region", col_prix="price"):
    print("\n=== Génération de la carte avec Stats Intégrées ===")
    
    # --- 1. MAPPING ET CALCULS (Code précédent) ---
    region_mapping = {
        "Alsace": "Grand Est", "Champagne-Ardenne": "Grand Est", "Lorraine": "Grand Est",
        "Aquitaine": "Nouvelle-Aquitaine", "Limousin": "Nouvelle-Aquitaine", "Poitou-Charentes": "Nouvelle-Aquitaine",
        "Auvergne": "Auvergne-Rhône-Alpes", "Rhône-Alpes": "Auvergne-Rhône-Alpes",
        "Bourgogne": "Bourgogne-Franche-Comté", "Franche-Comté": "Bourgogne-Franche-Comté",
        "Languedoc-Roussillon": "Occitanie", "Midi-Pyrénées": "Occitanie",
        "Nord-Pas-de-Calais": "Hauts-de-France", "Picardie": "Hauts-de-France",
        "Basse-Normandie": "Normandie", "Haute-Normandie": "Normandie",
        "Centre": "Centre-Val de Loire", "Ile-de-France": "Île-de-France",
        "Bretagne": "Bretagne", "Corse": "Corse", 
        "Pays de la Loire": "Pays de la Loire", "Provence-Alpes-Côte d'Azur": "Provence-Alpes-Côte d'Azur"
    }

    df_map = df.copy()
    df_map["region_actuelle"] = df_map[col_region].map(region_mapping).fillna(df_map[col_region])
    prix_moyen = df_map.groupby("region_actuelle")[col_prix].mean().reset_index()
    prix_moyen.columns = ["nom", "prix_moyen"]

    # --- 2. CALCUL DES STATISTIQUES POUR LA LÉGENDE ---
    # On récupère les lignes Min et Max
    row_max = prix_moyen.loc[prix_moyen["prix_moyen"].idxmax()]
    row_min = prix_moyen.loc[prix_moyen["prix_moyen"].idxmin()]
    mean_val = prix_moyen["prix_moyen"].mean()
    std_val = prix_moyen["prix_moyen"].std()

    # Détection automatique du mode (Loyer vs Achat) pour le formatage
    is_vente = mean_val > 10000
    unit = "k€" if is_vente else "€"
    
    def fmt(x):
        return f"{int(x/1000)}" if is_vente else f"{int(x)}"

    stats_text = (
        f"STATISTICS\n"
        f"────────────────\n"
        f"Mean : {fmt(mean_val)}{unit}\n"
        f"Std Dev : {fmt(std_val)}{unit}\n"
        f"Max : {row_max['nom']} ({fmt(row_max['prix_moyen'])}{unit})\n"
        f"Min : {row_min['nom']} ({fmt(row_min['prix_moyen'])}{unit})"
    )

    # --- 3. CHARGEMENT ET PLOT ---
    url_geojson = "https://github.com/gregoiredavid/france-geojson/raw/master/regions.geojson"
    try:
        gdf = gpd.read_file(url_geojson)
    except Exception: return

    merged = gdf.merge(prix_moyen, on="nom", how="left")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10)) # Un peu plus large
    
    merged.plot(column="prix_moyen", cmap="OrRd", linewidth=0.8, ax=ax, edgecolor="0.8",
                legend=True, legend_kwds={'label': f"Average Price ({unit})", 'orientation': "horizontal", "shrink": 0.5},
                missing_kwds={'color': 'lightgrey'})

    # Annotations sur la carte (Régions)
    merged["coords"] = merged["geometry"].apply(lambda x: x.representative_point().coords[:])
    merged["coords"] = [coords[0] for coords in merged["coords"]]
    for idx, row in merged.iterrows():
        if pd.notnull(row["prix_moyen"]):
            plt.annotate(text=f"{row['nom']}\n{fmt(row['prix_moyen'])}{unit}", 
                         xy=row['coords'], horizontalalignment='center', 
                         fontsize=8, color="black", fontweight="bold")

    # --- 4. AJOUT DE LA BOITE DE STATS ---
    # On place une boite de texte en bas à gauche (coordonnées relatives axes : 0.02, 0.02)
    plt.text(
        0.00, 0.00,             # Coordonnées (x, y) relatives à la figure (0 à 1)
        stats_text, 
        transform=ax.transAxes, 
        fontsize=10,
        verticalalignment='top', # Ancrage par le haut
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    )
    ax.set_axis_off()
    
    # Nom du fichier dynamique selon le type
    filename = "map_france_prix_achat.png" if is_vente else "map_france_prix_location.png"
    save_path = f"plots/{filename}"
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Carte avec stats sauvegardée : {save_path}")
    plt.close()
def debug_region_names(df, col_region="region"):
    """
    Outil de diagnostic pour comparer vos noms de régions avec ceux du GeoJSON officiel.
    """
    print("\n=== DIAGNOSTIC DES NOMS DE RÉGIONS ===")
    
    # 1. Charger le GeoJSON
    url = "https://github.com/gregoiredavid/france-geojson/raw/master/regions.geojson"
    try:
        gdf = gpd.read_file(url)
        print("✅ GeoJSON chargé avec succès.")
    except Exception as e:
        print(f"❌ Erreur chargement GeoJSON : {e}")
        return

    # 2. Récupérer les noms officiels (ce que le JSON "veut")
    # On regarde si la colonne s'appelle 'nom' ou 'name'
    if "nom" in gdf.columns:
        target_col = "nom"
    elif "name" in gdf.columns:
        target_col = "name"
    else:
        print(f"❌ Pas de colonne 'nom' trouvée. Colonnes dispos : {gdf.columns}")
        return

    official_names = sorted(gdf[target_col].unique())
    
    # 3. Récupérer vos noms (ce que vous avez)
    if col_region not in df.columns:
        print(f"❌ La colonne '{col_region}' n'existe pas dans votre DataFrame.")
        return
    your_names = sorted(df[col_region].dropna().unique())

    # 4. Affichage côte à côte
    print(f"\n{'VOS DONNÉES':<30} | {'GEOJSON OFFICIEL (ATTENDU)':<30}")
    print("-" * 65)
    
    # On affiche tout pour comparer visuellement
    max_len = max(len(official_names), len(your_names))
    for i in range(max_len):
        yours = your_names[i] if i < len(your_names) else ""
        official = official_names[i] if i < len(official_names) else ""
        
        # Marqueur visuel si correspondance exacte
        match = "✅" if yours in official_names else "⚠️"
        if yours == "": match = "" # Pas de marqueur si liste finie
        
        print(f"{yours:<30} {match} | {official}")

    # 5. Résumé des non-correspondances exactes
    print("\n--- Analyse des différences ---")
    missing = [n for n in your_names if n not in official_names]
    if missing:
        print(f"⚠️ Ces {len(missing)} régions de votre CSV ne trouvent pas de correspondance exacte :")
        for m in missing:
            print(f"   - '{m}'")
        print("\nConseil : Utilisez la fonction 'normalize_name' ou renommez-les dans votre CSV.")
    else:
        print("✅ Tout correspond parfaitement !")

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

    #analyse_numerique_numerique(df)
    #analyse_num_cat(df)
    #analyse_multilabels(df)
    #analyse_geo(df)
    analyse_anova(df)

    #debug_region_names(df)
    map_prix_par_region(df)




# Point d'entrée
if __name__ == "__main__":
    main()

