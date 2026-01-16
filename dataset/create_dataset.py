import os
import argparse
import pandas as pd
import time
import glob
from tqdm import tqdm
from functools import partial
import multiprocessing
import csv
import re



# =============================================================================
# 1. CONFIGURATION & DICTIONNAIRES
# =============================================================================

rename_dict = {
    # --- A. ABSOLUTE KILLERS (DROP DIRECT) ---
    "type_vente": "dataset_source",
    "prix": "price",
    "latitude": "latitude",
    "longitude": "longitude",
    "surface_habitable": "living_area_sqm",
    "type_bien": "property_type",
    "images_urls": "images",
    "id": "id",
    "titre": "titre",
    "description": "description",
    "nb_pieces": "num_rooms",
    "etat_bien": "property_status",

    # --- B. CONDITIONAL KILLERS (MAISON) ---
    "surface_tolale_terrain": "total_land_area_sqm", 

    # --- C. ACCEPTABLES (CHECK FOR QUALITY GATE) ---
    "classe_energetique": "energy_rating",
    "annee_construction": "year_built",
    "nb_chambres": "num_bedrooms",
    "nb_salleDeBains": "num_bathrooms",
    "orientation": "orientation",

    # --- D. LOGICAL FILL (NaN = 0 ou "") ---
    "nb_placesParking": "num_parking_spaces",
    "acces_exterieur": "exterior_access",
    "specificites": "special_features",

    # --- E. ANALYSIS ONLY ---
    "date_publication": "publication_date",
    "ville": "city",
    "region": "region",
    "codePostal": "postal_code",
    "departement": "departement",
    "frais_notaire_estimes": "estimated_notary_fees",
}

# --- LISTES AUTOMATIQUES ---
colonnes_cibles = list(rename_dict.values())

ABSOLUTE_KILLERS = [
    "dataset_source", "price", "latitude", "longitude", 
    "living_area_sqm", "property_type", "images", 
    "id", "titre", "description", "num_rooms", "property_status"
]

FILL_ZERO_COLS = ["num_parking_spaces"]

FILL_EMPTY_COLS = [
    "exterior_access", "special_features", "orientation",
    "publication_date", "city", "region", "postal_code", 
    "departement", "estimated_notary_fees"
]

# Colonnes surveillées par le QUALITY GATE (>2 manquants = Drop)
QUALITY_CHECK_COLS = [
    "energy_rating", 
    "year_built", 
    "num_bedrooms", 
    "num_bathrooms", 
    "orientation"
]

RULES_COMMON = {
    "num_bathrooms": (1, 20),
    "num_parking_spaces": (0, 50),
    "year_built": (1600, 2025)
}

RULES_APPART = {
    "num_rooms": (1, 20),
    "num_bedrooms": (0, 15),
    "living_area_sqm": (9, 600),
    "total_land_area_sqm": (0, 1000), 
}

RULES_MAISON = {
    "num_rooms": (1, 50),
    "num_bedrooms": (1, 30),
    "living_area_sqm": (20, 2000),
    "total_land_area_sqm": (1, 1000000),
}


# =============================================================================
# 2. NETTOYAGE TEXTE (REGEX)
# =============================================================================
SENTENCE_SPLITTER = re.compile(r'(?:[\.\!\?]\s+|\n+)')

BANNED_PATTERNS_LIST = [
    r'\d[\d\s\.,]*\s*(€|eur|euro|k€|\$|francs)', 
    r'\bprix\b', r'\bloyer\b', r'\bbudget\b', r'\bvaleur\b', r'\bmontant\b', r'\btarif\b', 
    r'\bsomme\b', r'\bcoût\b', r'\bfinance', r'\bpaiement\b', r'\bcrédit\b', r'\bmensualité\b', 
    r'\brentabilité\b', r'\bca\b', r'\bhonoraires\b', r'\bfrais\s*d\'?agence\b', r'\bfai\b', 
    r'\bhai\b', r'\bttc\b', r'\bht\b', r'\bnet\s*vendeur\b', r'\bcharges\b', 
    r'\bdepot\s*de\s*garantie\b', r'\bcaution\b', r'\bnotaire\b', r'\btaxe\s*foncière\b', 
    r'\btaxe\s*habitation\b', r'\bcc\b', r'\bhc\b', r'\bmensuel\b', r'/\s*mois', r'par\s*mois', 
    r'\btransaction\b', r'\binvestissement\b', r'\binvestisseur\b', r'\bnégociable\b', 
    r'\bfaire\s*offre\b', r'\bnous\s*consulter\b', r'\bestimation\b', r'\bvendue?\b', 
    r'\blouée?\b', r'(tél|tel|phone|mobile|port|contact)\b', 
    r'[0-9]{2}[\s\.]?[0-9]{2}[\s\.]?[0-9]{2}', r'exclusivite', r'exclusivité', r'[@]', 
    r'www\.', r'\[coordonnées masquées\]', r'géorisques', r'loi alur', r'copropriété', 
    r'syndic', r'procédure en cours', r'rsac', r'siret', r'référence\s*:', r'visite', 
    r'\b(nb|nombre|copropriété)\s*(de)?\s*(\d+\s*lots?|lots?\s*[:\s\.]*\d+)'
]
BANNED_REGEX = re.compile('|'.join(BANNED_PATTERNS_LIST))


def clean_description_text(text):
    if not isinstance(text, str): return ""
    text = text.lower().replace('\xa0', ' ').replace('\r', ' ')
    sentences = SENTENCE_SPLITTER.split(text)
    kept_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue
        if BANNED_REGEX.search(sentence): continue
        kept_sentences.append(sentence)
    clean_text = ". ".join(kept_sentences)
    return re.sub(r'\.+', '.', clean_text).strip()


def apply_rules(df, rules):
    """Applique les bornes Min/Max sans supprimer les NaN."""
    for col, (mini, maxi) in rules.items():
        if col in df.columns:
            s_numeric = pd.to_numeric(df[col], errors='coerce')
            mask = s_numeric.between(mini, maxi) | s_numeric.isna()
            df = df[mask].copy()
            try:
                df[col] = df[col].astype("Int64") 
            except:
                pass
    return df


# =============================================================================
# 3. FILTRE PRINCIPAL
# =============================================================================

def filtre(df):
    if df is None or df.empty: return df

    # --- ÉTAPE 1 : PROTECTION & REMPLISSAGE LOGIQUE ---
    for col in FILL_ZERO_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col in FILL_EMPTY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # --- ÉTAPE 2 : ABSOLUTE KILLERS ---
    current_killers = [c for c in ABSOLUTE_KILLERS if c in df.columns]
    df = df.dropna(subset=current_killers)
    if df.empty: return df

    # --- ÉTAPE 3 : FILTRAGE TYPE DE BIEN ---
    df = df[df['property_type'].isin(['Appartement', 'Maison'])]
    if df.empty: return df

    # --- ÉTAPE 4 : SPLIT & LOGIQUE CONDITIONNELLE ---
    mask_appart = df['property_type'] == 'Appartement'
    mask_maison = df['property_type'] == 'Maison'

    df_appart = df[mask_appart].copy()
    df_maison = df[mask_maison].copy()

    # Maison : Terrain obligatoire
    df_maison = df_maison.dropna(subset=["total_land_area_sqm"])
    # Appart : Terrain = 0
    df_appart["total_land_area_sqm"] = df_appart["total_land_area_sqm"].fillna(0)

    # --- ÉTAPE 5 : RÈGLES DE VALEURS ---
    df_appart = apply_rules(df_appart, RULES_COMMON)
    df_appart = apply_rules(df_appart, RULES_APPART)

    df_maison = apply_rules(df_maison, RULES_COMMON)
    df_maison = apply_rules(df_maison, RULES_MAISON)

    # --- ÉTAPE 6 : FUSION ---
    df_final = pd.concat([df_appart, df_maison], ignore_index=True)
    if df_final.empty: return df_final

    # --- ÉTAPE 7 : NETTOYAGE DPE (SANITIZATION) ---
    df_final["energy_rating"] = df_final["energy_rating"].astype(str).str.upper().str.strip()
    valid_ratings = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    # Si invalide ou bizarre, on remplace par "" (Vide)
    df_final.loc[~df_final["energy_rating"].isin(valid_ratings), "energy_rating"] = ""

    # --- ÉTAPE 8 : QUALITY GATE (MAX 2 MANQUANTS) ---
    # On vérifie la qualité des lignes restantes
    check_cols = [c for c in QUALITY_CHECK_COLS if c in df_final.columns]
    
    if check_cols:
        # On compte comme manquant : NaN (pd.NA) OU Vide ("")
        # Note: energy_rating propre est soit "A" soit "" à ce stade.
        missing_mask = df_final[check_cols].isna() | (df_final[check_cols] == "")
        missing_count = missing_mask.sum(axis=1)

        # REGLE STRICTE : On garde si manquants <= 2
        df_final = df_final[missing_count <= 2]
        
    if df_final.empty: return df_final

    # --- ÉTAPE 9 : FILTRES FINAUX ---
    # Images (Min 3)
    images_str = df_final['images'].astype(str)
    num_images = images_str.str.count(r'\|') + (images_str != '').astype(int)
    df_final = df_final[num_images >= 3]

    # Description Clean
    df_final['description'] = df_final['description'].astype(str).apply(clean_description_text)
    df_final = df_final[df_final['description'].str.len() > 25]

    return df_final.drop_duplicates(subset=['id'], keep='last', ignore_index=True)


# =============================================================================
# 4. EXECUTION
# =============================================================================

def worker(data_package, output_dir):
    csv_path, source_type = data_package
    try:
        df = pd.read_csv(csv_path, sep=";", quotechar='"', low_memory=False)
        if df.empty: return source_type, 0, 0

        df.rename(columns=rename_dict, inplace=True)
        cols_to_keep = [c for c in colonnes_cibles if c in df.columns]
        df = df[cols_to_keep]

        nb_avant = len(df)
        df_filtered = filtre(df)
        nb_apres = len(df_filtered)

        if nb_apres > 0:
            final_output_dir = os.path.join(output_dir, source_type.lower())
            output_file = os.path.join(final_output_dir, os.path.basename(csv_path))
            df_filtered.to_csv(output_file, sep=";", index=False, quoting=csv.QUOTE_MINIMAL)

        return source_type, nb_avant, nb_apres

    except Exception as e:
        return source_type, 0, 0


def run(args):
    t0 = time.monotonic()
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'achat'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'location'), exist_ok=True)

    files_achat = [(f, 'ACHAT') for f in glob.glob(os.path.join(args.achat, '*.csv'))]
    files_loc = [(f, 'LOCATION') for f in glob.glob(os.path.join(args.location, '*.csv'))]
    all_tasks = files_achat + files_loc

    if not all_tasks:
        print(f"[WARN] Aucun fichier CSV trouvé.")
        return

    num_processes = min(args.workers, len(all_tasks))
    print(f"Traitement de {len(all_tasks)} fichiers ({len(files_achat)} Achat, {len(files_loc)} Location)...")

    stats = {'ACHAT': {'avant': 0, 'apres': 0}, 'LOCATION': {'avant': 0, 'apres': 0}}
    
    fun = partial(worker, output_dir=args.output)
    with multiprocessing.Pool(processes=num_processes) as pool:
        for source_type, avant, apres in tqdm(pool.imap_unordered(fun, all_tasks), total=len(all_tasks)):
            stats[source_type]['avant'] += avant
            stats[source_type]['apres'] += apres

    dt = time.monotonic() - t0

    print("\n" + "="*65)
    print(f"{'SOURCE':<10} | {'AVANT':<12} | {'APRÈS':<12} | {'SUPPRIMÉS':<15}")
    print("-" * 65)
    tot_avant, tot_apres = 0, 0
    for key in ['ACHAT', 'LOCATION']:
        av, ap = stats[key]['avant'], stats[key]['apres']
        suppr = av - ap
        pct = (suppr / av * 100) if av > 0 else 0
        tot_avant += av; tot_apres += ap
        print(f"{key:<10} | {av:<12,} | {ap:<12,} | {suppr:<9,} (-{pct:.1f}%)".replace(",", " "))
    print("-" * 65)
    tot_suppr = tot_avant - tot_apres
    tot_pct = (tot_suppr / tot_avant * 100) if tot_avant > 0 else 0
    print(f"{'TOTAL':<10} | {tot_avant:<12,} | {tot_apres:<12,} | {tot_suppr:<9,} (-{tot_pct:.1f}%)".replace(",", " "))
    print("="*65)
    print(f"Temps total : {dt:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--achat', type=str, default='../../data/achat/')
    parser.add_argument('-l', '--location', type=str, default='../../data/location/')
    parser.add_argument('-o', '--output', type=str, default='../output/raw_csv')
    parser.add_argument('-w', '--workers', type=int, default=max(1, multiprocessing.cpu_count()-1))
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()
