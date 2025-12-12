import os
import argparse
import pandas as pd
import time
import glob
from multiprocessing import Pool, cpu_count
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns



# --- CONFIGURATION GRAPHIQUE ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'


def count_images_recursively(root_dir):
    """
    Parcourt récursivement les dossiers.
    Compte TOUS les fichiers trouvés (supposés être des images).
    """
    if not os.path.exists(root_dir): 
        return 0
    
    count = 0
    # os.walk est très efficace pour parcourir une arborescence complexe
    for root, dirs, files in os.walk(root_dir):
        count += len(files)

    return count


def get_chunk_stats(file_list):
    """Worker : Traite un lot de fichiers CSV"""
    if not file_list: return None
    dfs = []
    for csv_path in file_list:
        try:
            # low_memory=False pour éviter les Warnings sur les types mixtes
            dfs.append(pd.read_csv(csv_path, sep=";", quotechar='"', low_memory=False))
        except Exception:
            pass 

    if not dfs: return None
    df_chunk = pd.concat(dfs, ignore_index=True)

    # 1. Structure de base
    total_rows = len(df_chunk)
    nan_counts = df_chunk.isna().sum()

    # 2. Compte par Type de Bien (Maison, Appartement, etc.)
    property_type_counts = df_chunk['type_bien'].value_counts()
    grouped_nan_counts = df_chunk.isna().groupby(df_chunk['type_bien']).sum()

    # 3. Compte par Type de Vente (Achat/Location)
    type_vente_counts = df_chunk['type_vente'].value_counts()

    # 4. Stats Numériques (Prix, Surface, Pièces)
    numeric_stats = {}
    cols_to_check = ['price', 'surface_area', 'rooms']
    for col in cols_to_check:
        if col in df_chunk.columns:
            s = pd.to_numeric(df_chunk[col], errors='coerce')
            numeric_stats[col] = {
                'min': s.min(), 'max': s.max(), 'sum': s.sum(), 'count': s.count()
            }
        else:
            numeric_stats[col] = None

    return {
        'total_rows': total_rows,
        'nan_counts': nan_counts,
        'grouped_nan_counts': grouped_nan_counts,
        'property_type_counts': property_type_counts,
        'type_vente_counts': type_vente_counts,
        'numeric_stats': numeric_stats
    }

def merge_stats(stat_a, stat_b):
    """Reducer : Fusionne les résultats des workers"""
    if stat_a is None: return stat_b
    if stat_b is None: return stat_a

    # Fusion des stats numériques
    merged_numeric = {}
    all_keys = set(stat_a['numeric_stats'].keys()) | set(stat_b['numeric_stats'].keys())
    for col in all_keys:
        val_a, val_b = stat_a['numeric_stats'].get(col), stat_b['numeric_stats'].get(col)
        if val_a and val_b:
            merged_numeric[col] = {
                'min': min(val_a['min'], val_b['min']),
                'max': max(val_a['max'], val_b['max']),
                'sum': val_a['sum'] + val_b['sum'],
                'count': val_a['count'] + val_b['count']
            }
        else:
            merged_numeric[col] = val_a if val_a else val_b

    return {
        'total_rows': stat_a['total_rows'] + stat_b['total_rows'],
        'nan_counts': stat_a['nan_counts'].add(stat_b['nan_counts'], fill_value=0),
        'grouped_nan_counts': stat_a['grouped_nan_counts'].add(stat_b['grouped_nan_counts'], fill_value=0),
        'property_type_counts': stat_a['property_type_counts'].add(stat_b['property_type_counts'], fill_value=0),
        'type_vente_counts': stat_a['type_vente_counts'].add(stat_b['type_vente_counts'], fill_value=0),
        'numeric_stats': merged_numeric
    }

def generate_plots(df_density, type_counts, out_dir):
    """Génère les graphiques pour le rapport"""
    print("   [INFO] Génération des graphiques...")
    
    # 1. Barplot manquants
    plt.figure(figsize=(12, 10))
    missing_data = df_density[df_density['pct_missing'] > 0].sort_values('pct_missing', ascending=True)
    
    if not missing_data.empty:
        chart = sns.barplot(
            data=missing_data,
            x='pct_missing', 
            y=missing_data.index, 
            palette='viridis',
            hue=missing_data.index,
            legend=False
        )
        chart.bar_label(chart.containers[0], fmt='%.1f%%', padding=3)
        plt.title('Qualité des Données : % Valeurs Manquantes', fontsize=15, weight='bold')
        plt.xlabel('% Manquant')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'plot_missing_values.png'), dpi=300)
        plt.close()

    # 2. Donut Chart Répartition
    plt.figure(figsize=(8, 8))
    top_types = type_counts.sort_values(ascending=False).head(5)
    others = type_counts.sum() - top_types.sum()
    if others > 0:
        top_types['Autres'] = others
    
    plt.pie(top_types, labels=top_types.index, autopct='%1.1f%%', startangle=140, 
            colors=sns.color_palette('pastel'), pctdistance=0.85, explode=[0.05] + [0]* (len(top_types)-1))
    
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.title('Répartition des Types de Biens', fontsize=15, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plot_property_types.png'), dpi=300)
    plt.close()

def run(args):
    t0 = time.monotonic()
    
    # Récupération de tous les fichiers
    files = glob.glob(os.path.join(args.acaht_csv, '*.csv')) + glob.glob(os.path.join(args.loc_csv, '*.csv'))
    
    if not files:
        print("[ERR] Aucun fichier CSV trouvé.")
        return

    # Multiprocessing
    print(f"Traitement de {len(files)} fichiers avec {args.workers} workers...")
    with Pool(processes=args.workers) as pool:
        chunk_size = (len(files) // args.workers) + 1
        chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
        results = pool.map(get_chunk_stats, chunks)

    final = reduce(merge_stats, results)
    
    if not final:
        print("[ERR] Échec du traitement.")
        return

    total_rows = final['total_rows']
    
    # --- EXTRACTION DES COMPTEURS SPÉCIFIQUES ---
    
    # 1. Location (1) vs Achat (0)
    tv_counts = final['type_vente_counts']
    nb_achat = tv_counts.get(0, 0) # Mapping: 0 = Achat
    nb_loc = tv_counts.get(1, 0)   # Mapping: 1 = Location
    
    # 2. Maison vs Appartement
    pt_counts = final['property_type_counts']
    # On cherche les clés 'Maison' et 'Appartement' (sensible à la casse dans le CSV)
    # On utilise .get() pour éviter un crash si le type n'existe pas
    nb_maison = pt_counts.get('Maison', 0)
    nb_appart = pt_counts.get('Appartement', 0)
    
    # Si les clés sont en minuscules dans ton CSV, on peut tenter une somme plus robuste :
    # nb_maison = pt_counts[pt_counts.index.str.lower() == 'maison'].sum()

    # --- AFFICHAGE DU RAPPORT ---
    print("\n" + "="*50)
    print(f" RAPPORT DATASET IMMOBILIER (N={total_rows:,})")
    print("="*50)

    print("\n--- RÉPARTITION TRANSACTION (Type Vente) ---")
    print(f"ACHAT (0)     : {int(nb_achat):,} ({nb_achat/total_rows:.1%})")
    print(f"LOCATION (1)  : {int(nb_loc):,} ({nb_loc/total_rows:.1%})")

    print("\n--- RÉPARTITION TYPE DE BIEN ---")
    print(f"MAISONS       : {int(nb_maison):,} ({nb_maison/total_rows:.1%})")
    print(f"APPARTEMENTS  : {int(nb_appart):,} ({nb_appart/total_rows:.1%})")
    # Afficher le reste du top 5 pour info
    print("-" * 30)
    print("Top 5 complet des types :")
    print(pt_counts.sort_values(ascending=False).head(5).to_string())

    print("\n--- QUALITÉ DES DONNÉES (% Manquant) ---")
    df_density = (final['nan_counts'] / total_rows * 100).to_frame(name='pct_missing')
    pd.options.display.max_rows = 100
    # On affiche tout trié
    print(df_density.sort_values('pct_missing', ascending=False).round(2))

    # --- SAUVEGARDE & PLOTS ---
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    
    df_density.to_csv(os.path.join(out_dir, 'stats_missing_pct.csv'), sep=';', float_format='%.2f')
    
    try:
        generate_plots(df_density, pt_counts, out_dir)
        print(f"\n[OK] Graphiques sauvegardés dans : {out_dir}")
    except Exception as e:
        print(f"\n[WARN] Erreur graphiques : {e}")

    # --- COMPTAGE IMAGES ---
    if args.img_dir:
        t_img = time.monotonic()
        nb_img = count_images_recursively(args.img_dir)
        print(f"\n--- MÉDIAS ---")
        print(f"Total Images (fichiers) : {nb_img:,}")
        print(f"Ratio Images/Annonce    : {nb_img/total_rows:.2f}")
        print(f"(Temps scan images: {time.monotonic() - t_img:.2f}s)")

    print(f"\nTemps total script : {time.monotonic() - t0:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--acaht_csv', type=str, default='../../data/achat/')
    parser.add_argument('-l', '--loc_csv', type=str, default='../../data/location/')
    parser.add_argument('-img', '--img_dir', type=str, default=None)
    parser.add_argument('-o', '--output', type=str, default='../../output/')
    parser.add_argument('-w', '--workers', type=int, default=max(1, cpu_count()-1))
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
