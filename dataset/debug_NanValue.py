import os
import glob
import pandas as pd
from tqdm import tqdm



rename_dict = {
    "id": "id",
    "titre": "titre",
    "type_bien": "property_type", 
    "etat_bien": "property_status",
    "prix": "price",
    "ville": "city",
    "region": "region",
    "codePostal": "postal_code",
    "departement": "department",
    "nb_pieces": "num_rooms", 
    "nb_chambres": "num_bedrooms", 
    "nb_salleDeBains": "num_bathrooms", 
    "classe_energetique": "energy_rating", 
    "orientation": "orientation", 
    "nb_placesParking": "num_parking_spaces", 
    "surface_habitable": "living_area_sqm", 
    "surface_tolale_terrain": "total_land_area_sqm", 
    "nb_etages_Immeuble": "building_num_floors", 
    #"prix_metre_carre": "price_per_sqm", 
    "annee_construction": "year_built", 
    "specificites": "features", 
    "images_urls": "images", 
    "description": "description",
}

# Dossiers (Ajuste les chemins si besoin)
LOC_DIR = '../../data/achat/'


def get_appart_stats():
    # 1. Récupérer tous les fichiers CSV
    files = glob.glob(os.path.join(LOC_DIR, '*.csv'))

    print(f"Analyse de {len(files)} fichiers pour trouver les problèmes sur les NAN et les vars associées...")

    all_apparts = []

    # 2. Chargement brut sans dropna
    for f in tqdm(files):
        try:
            # On lit tout en string pour ne pas avoir d'erreur de conversion
            df = pd.read_csv(f, sep=";", quotechar='"', dtype=str)
            
            # Renommage
            df.rename(columns=rename_dict, inplace=True)
            df["total_land_area_sqm"] = df["total_land_area_sqm"].fillna(0)
            df["num_parking_spaces"] = df["num_parking_spaces"].fillna(0)
            df["features"] = df["features"].fillna("rien")

            # On filtre uniquement sur le texte "appart" dans le type
            # (On gère les NaN dans property_type pour ne pas planter)
            if 'property_type' in df.columns:
                mask = df['property_type'].astype(str).str.lower().str.contains('appart', na=False)
                df_appart = df[mask]
                
                # On garde les colonnes cibles uniquement pour la lisibilité
                cols_to_keep = [c for c in rename_dict.values() if c in df_appart.columns]
                all_apparts.append(df_appart[cols_to_keep])
                
        except Exception as e:
            print(f"Erreur lecture {f}: {e}")

    # 3. Concaténation
    if not all_apparts:
        print("AUCUN appartement trouvé dans les fichiers bruts !")
        return

    full_df = pd.concat(all_apparts, ignore_index=True)
    total_rows = len(full_df)

    print("\n" + "="*50)
    print(f"TOTAL APPARTEMENTS BRUTS TROUVÉS : {total_rows}")
    print("="*50)

    # 4. Analyse des NaN et Vides
    # Comme on a chargé en string, les vides peuvent être NaN ou "" (chaine vide)
    print(f"{'COLONNE':<25} | {'MANQUANTS (NaN + Vide)':<20} | {'% MANQUANT'}")
    print("-" * 60)

    stats = []
    
    for col in full_df.columns:
        # Compte les NaN
        nb_nan = full_df[col].isna().sum()
        # Compte les chaines vides "" ou "nan" texte
        nb_empty_str = full_df[col].astype(str).str.strip().isin(['', 'nan', 'NaN']).sum()
        
        # Le total des valeurs "inutilisables"
        # (Attention : isna() est inclus dans le check string souvent si converti, 
        # mais on prend le max ou la somme logique pour être sûr)
        nb_missing = nb_nan + (nb_empty_str - nb_nan if nb_nan < nb_empty_str else 0) 
        
        pct = (nb_missing / total_rows) * 100
        stats.append((col, nb_missing, pct))

    # Tri par le pire coupable (celui qui a le plus de manquants)
    stats.sort(key=lambda x: x[1], reverse=True)

    for col, nb, pct in stats:
        bar = "|" * int(pct / 5)
        print(f"{col:<25} | {nb:<20} | {pct:6.2f}% {bar}")

    print("-" * 60)
    print("CONSEIL : Si une colonne essentielle dépasse 0%, le 'dropna()' supprime ces lignes.")

if __name__ == "__main__":
    get_appart_stats()
