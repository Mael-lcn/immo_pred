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



rename_dict = {
    "id": "id",
    "titre": "titre",
    "type_vente": "dataset_source",
    "type_bien": "property_type", 
    "etat_bien": "property_status",
    "prix": "price",
    "latitude": "latitude",
    "longitude": "longitude",
    "nb_pieces": "num_rooms", 
    "nb_chambres": "num_bedrooms", 
    "nb_salleDeBains": "num_bathrooms", 
    "classe_energetique": "energy_rating", 
    "orientation": "orientation", 
    "nb_placesParking": "num_parking_spaces", 
    "surface_habitable": "living_area_sqm", 
    "surface_tolale_terrain": "total_land_area_sqm", 
    "annee_construction": "year_built", 
    "acces_exterieur": "exterior_access",
    "specificites": "special_features", 
    "images_urls": "images",
    "description": "description",

    #####
    # Only for the annalysis use fillna("")
    #####
    "date_publication": "publication_date",
    "ville": "city",
    "region": "region",
    "codePostal": "postal_code",
    "departement": "departement",
    "frais_notaire_estimes": "estimated_notary_fees",
}

colonnes_cibles = list(rename_dict.values())



# --- RÈGLES DE FILTRAGE ---
RULES_COMMON = {
    "num_bathrooms": (1, 20),
    "num_parking_spaces": (0, 50),
    "price_per_sqm": (200, 60000),
    "year_built": (1600, 2025)
}

RULES_APPART = {
    "num_rooms": (1, 20),
    "num_bedrooms": (1, 15),
    "living_area_sqm": (10, 600),
    "total_land_area_sqm": (0, 1000),
    "building_num_floors": (0, 35)
}

RULES_MAISON = {
    "num_rooms": (1, 50),
    "num_bedrooms": (1, 30),
    "living_area_sqm": (20, 1500),
    "total_land_area_sqm": (0, 800000),
    "building_num_floors": (0, 4)
}



# =============================================================================
# 1. PRÉ-COMPILATION DES REGEX (Exécuté 1 seule fois au démarrage)
# =============================================================================

# A. Découpeur de phrases
# On coupe sur : Points/Exclamation/Interrogation suivis d'espace, ou Sauts de ligne
SENTENCE_SPLITTER = re.compile(r'(?:[\.\!\?]\s+|\n+)')

# B. Motifs Interdits
# Si une phrase contient un de ces motifs, ELLE EST ENTIÈREMENT SUPPRIMÉE.
BANNED_PATTERNS_LIST = [
    # =========================================================================
    # 1. DÉTECTION MONÉTAIRE DIRECTE (Le Leakage Absolu)
    # =========================================================================
    # Capture : "400 €", "400.000 euros", "1200 EUR", "500k€", "400 $"
    # \d[\d\s\.,]* : Un chiffre suivi potentiellement d'espaces, points ou virgules
    r'\d[\d\s\.,]*\s*(€|eur|euro|k€|\$|francs)', 

    # =========================================================================
    # 2. MOTS-CLÉS "ARGENT" EXPLICITES
    # =========================================================================
    r'\bprix\b',                 # "Prix de vente", "Prix du loyer"
    r'\bloyer\b',                # "Loyer mensuel", "Appel de loyer"
    r'\bbudget\b',               # "Budget à prévoir", "Petit budget"
    r'\bvaleur\b',               # "Valeur locative", "Valeur du bien"
    r'\bmontant\b',              # "Montant des charges"
    r'\btarif\b',                # "Tarif attractif"
    r'\bsomme\b',                # "Somme demandée"
    r'\bcoût\b',                 # "Coût total", "Coût énergie"
    r'\bfinance',                # "Financement", "Financier"
    r'\bpaiement\b',             # "Facilités de paiement"
    r'\bcrédit\b',               # "Crédit vendeur", "Crédit immobilier"
    r'\bmensualité\b',           # "Mensualités de..."
    r'\brentabilité\b',          # "Forte rentabilité" (Indice fort sur le prix)
    r'\bca\b',                   # "Chiffre d'affaires" (pour les locaux comm.)

    # =========================================================================
    # 3. FRAIS, CHARGES ET JARGON D'AGENCE
    # =========================================================================
    r'\bhonoraires\b',           # "Honoraires charge vendeur/acquéreur"
    r'\bfrais\s*d\'?agence\b',   # "Frais d'agence"
    r'\bfai\b',                  # "Frais Agence Inclus"
    r'\bhai\b',                  # "Honoraires Agence Inclus"
    r'\bttc\b',                  # "Toutes Taxes Comprises" (Souvent collé au prix)
    r'\bht\b',                   # "Hors Taxe"
    r'\bnet\s*vendeur\b',        # "Prix net vendeur"
    r'\bcharges\b',              # "Charges de copro", "Charges mensuelles"
    r'\bdepot\s*de\s*garantie\b',# "Dépôt de garantie"
    r'\bcaution\b',              # "Caution solidaire", "Caution de X euros"
    r'\bnotaire\b',              # "Frais de notaire"
    r'\btaxe\s*foncière\b',      # Souvent donnée avec le prix
    r'\btaxe\s*habitation\b',
    r'\bcc\b',                   # "Loyer CC" (Charges Comprises)
    r'\bhc\b',                   # "Loyer HC" (Hors Charges)

    # =========================================================================
    # 4. TERMES TEMPORELS LIÉS À L'ARGENT
    # =========================================================================
    r'\bmensuel\b',              # "Loyer mensuel", "Paiement mensuel"
    r'/\s*mois',                 # "500 / mois"
    r'par\s*mois',               # "500 par mois"

    # =========================================================================
    # 5. TERMES DE TRANSACTION & NÉGOCIATION (Contexte dangereux)
    # =========================================================================
    r'\btransaction\b',
    r'\binvestissement\b',       # "Idéal investissement" (Biais le modèle vers le bas prix)
    r'\binvestisseur\b',         # "Idéal investisseur"
    r'\bnégociable\b',           # "Prix négociable"
    r'\bfaire\s*offre\b',        # "Faire offre raisonnable"
    r'\bnous\s*consulter\b',     # "Prix : nous consulter"
    r'\bestimation\b',           # "Estimation gratuite"
    r'\bvendue?\b',              # "Déjà vendu" (Noise)
    r'\blouée?\b',               # "Déjà loué"

    # =========================================================================
    # 6. CONTACTS & BRUIT ADMINISTRATIF (Rappel de sécurité)
    # =========================================================================
    r'(tél|tel|phone|mobile|port|contact)\b',
    r'[0-9]{2}[\s\.]?[0-9]{2}[\s\.]?[0-9]{2}', # 06 00...
    r'exclusivite',
    r'exclusivité',
    r'[@]',
    r'www\.',
    r'\[coordonnées masquées\]',

    # Administratif strict
    r'géorisques',
    r'loi alur',
    r'copropriété',
    r'syndic',
    r'procédure en cours',
    r'rsac',
    r'siret',
    r'référence\s*:',
    r'visite',
    r'\b(nb|nombre|copropriété)\s*(de)?\s*(\d+\s*lots?|lots?\s*[:\s\.]*\d+)',]

# Compilation du motif géant optimisé
# On joint tout avec OU (|)
BANNED_REGEX = re.compile('|'.join(BANNED_PATTERNS_LIST))


# =============================================================================
# 2. FONCTION DE NETTOYAGE
# =============================================================================

def clean_description_text(text):
    """
    Version ultra-rapide : utilise les regex pré-compilées globales.
    """
    if not isinstance(text, str):
        return ""

    # 1. Normalisation
    # On passe en minuscule tout de suite pour matcher les regex
    text = text.lower().replace('\xa0', ' ').replace('\r', ' ')

    # 2. Découpage (Utilise l'objet compilé global)
    sentences = SENTENCE_SPLITTER.split(text)

    kept_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: 
            continue

        # 3. Filtrage (Utilise l'objet compilé global)
        if BANNED_REGEX.search(sentence):
            continue

        kept_sentences.append(sentence)

    # 4. Reconstruction
    clean_text = ". ".join(kept_sentences)
    clean_text = re.sub(r'\.+', '.', clean_text) # Nettoyage rapide des points multiples

    return clean_text.strip()


def apply_rules(df, rules):
    for col, (mini, maxi) in rules.items():
        if col in df.columns:
            s_numeric = pd.to_numeric(df[col], errors='coerce')
            mask = s_numeric.between(mini, maxi)
            df = df[mask]
            df[col] = df[col].astype(float).astype(int)

    return df


def filtre(df):
    if df is None or df.empty:
        return df

    # Nettoyage de base
    df = df.dropna(subset=['property_type'])
    df["total_land_area_sqm"] = df["total_land_area_sqm"].fillna(0)
    df["num_parking_spaces"] = df["num_parking_spaces"].fillna(0)
    df["special_features"] = df["special_features"].fillna("")
    df["exterior_access"] = df["exterior_access"].fillna("")


    annalysis_only = ["publication_date", "city", "region", "postal_code", "departement", "estimated_notary_fees"]
    for var in annalysis_only:
        df[var] = df[var].fillna("")

    df = df.dropna()
    df = df[df['property_type'].isin(['Appartement', 'Maison'])]

    # Règles communes
    df = apply_rules(df, RULES_COMMON)
    if df.empty: return df

    # Séparation Appart / Maison
    type_series = df['property_type'].astype(str).str.lower()
    mask_appart = type_series.str.contains('appartement')
    mask_maison = type_series.str.contains('maison')

    df_appart = df[mask_appart]
    df_maison = df[mask_maison]

    # Règles spécifiques
    df_appart = apply_rules(df_appart, RULES_APPART)
    df_maison = apply_rules(df_maison, RULES_MAISON)

    # Fusion
    df_final = pd.concat([df_appart, df_maison], ignore_index=True)
    if df_final.empty: return df_final

    # Filtres finaux (DPE et Images)
    valid_ratings = ['A', 'B', 'C', 'D', 'E', 'F']
    df_final["energy_rating"] = df_final["energy_rating"].astype(str).str.upper().str.strip()
    df_final = df_final[df_final["energy_rating"].isin(valid_ratings)]

    images_str = df_final['images'].astype(str)
    num_images = images_str.str.count(r'\|') + (images_str != '').astype(int)
    df_final = df_final[num_images >= df_final['num_rooms']]

    # --- APPLICATION DU NETTOYAGE TEXTE ---
    # On applique la fonction définie plus haut sur toute la colonne description
    df_final['description'] = df_final['description'].astype(str).apply(clean_description_text)
    # On retire les lignes où la description est devenue vide ou trop courte
    df_final = df_final[df_final['description'].str.len() > 25]

    return df_final.drop_duplicates(ignore_index=True)


def worker(data_package, output_dir):
    """
    data_package est un tuple : (chemin_du_fichier, type_source)
    type_source vaut soit 'ACHAT' soit 'LOCATION'
    """
    csv_path, source_type = data_package

    try:
        # Lecture
        df = pd.read_csv(csv_path, sep=";", quotechar='"')
        if df.empty: return source_type, 0, 0

        # Renommage
        df.rename(columns=rename_dict, inplace=True)
        cols_presentes = [c for c in colonnes_cibles if c in df.columns]
        df = df[cols_presentes]

        nb_avant = len(df)

        # Filtrage (inclut maintenant le nettoyage texte)
        df_filtered = filtre(df)

        nb_apres = len(df_filtered)

        # Écriture si données restantes
        if nb_apres > 0:
            final_output_dir = os.path.join(output_dir, source_type.lower())
            output_file = os.path.join(final_output_dir, os.path.basename(csv_path))

            df_filtered.to_csv(output_file, sep=";", index=False, quoting=csv.QUOTE_MINIMAL)

        return source_type, nb_avant, nb_apres

    except Exception as e:
        print(f"[ERROR] {csv_path} : {e}")
        return source_type, 0, 0


def run(args):
    t0 = time.monotonic()
    
    # Création des dossiers parents et enfants
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'achat'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'location'), exist_ok=True)

    # 1. On crée une liste de tuples (Fichier, Type)
    files_achat = [(f, 'ACHAT') for f in glob.glob(os.path.join(args.achat, '*.csv'))]
    files_loc = [(f, 'LOCATION') for f in glob.glob(os.path.join(args.location, '*.csv'))]

    all_tasks = files_achat + files_loc

    if not all_tasks:
        print(f"[WARN] Aucun fichier CSV trouvé.")
        return

    num_processes = min(args.workers, len(all_tasks))
    print(f"Traitement de {len(all_tasks)} fichiers ({len(files_achat)} Achat, {len(files_loc)} Location)...")

    # Dictionnaire pour compter séparément
    stats = {
        'ACHAT': {'avant': 0, 'apres': 0},
        'LOCATION': {'avant': 0, 'apres': 0}
    }

    fun = partial(worker, output_dir=args.output)

    with multiprocessing.Pool(processes=num_processes) as pool:
        for source_type, avant, apres in tqdm(pool.imap_unordered(fun, all_tasks), total=len(all_tasks)):
            stats[source_type]['avant'] += avant
            stats[source_type]['apres'] += apres

    dt = time.monotonic() - t0

    # --- AFFICHAGE DU TABLEAU COMPARATIF ---
    print("\n" + "="*65)
    print(f"{'SOURCE':<10} | {'AVANT':<12} | {'APRÈS':<12} | {'SUPPRIMÉS':<15}")
    print("-" * 65)

    tot_avant = 0
    tot_apres = 0

    for key in ['ACHAT', 'LOCATION']:
        av = stats[key]['avant']
        ap = stats[key]['apres']
        suppr = av - ap
        pct = (suppr / av * 100) if av > 0 else 0
        
        tot_avant += av
        tot_apres += ap
        
        print(f"{key:<10} | {av:<12,} | {ap:<12,} | {suppr:<9,} (-{pct:.1f}%)".replace(",", " "))

    print("-" * 65)
    tot_suppr = tot_avant - tot_apres
    tot_pct = (tot_suppr / tot_avant * 100) if tot_avant > 0 else 0

    print(f"{'TOTAL':<10} | {tot_avant:<12,} | {tot_apres:<12,} | {tot_suppr:<9,} (-{tot_pct:.1f}%)".replace(",", " "))
    print("="*65)
    print(f"Temps total : {dt:.2f}s")
    print(f"Sortie Achat    : {os.path.join(args.output, 'achat')}")
    print(f"Sortie Location : {os.path.join(args.output, 'location')}")



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

