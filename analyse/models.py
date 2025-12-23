import argparse
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from outils import load_all_regions,enleve_luxe
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


def residuals_plot(y_test, y_pred_test_final,type="xgb"):

    # Calcul des résidus (La différence entre Réel et Prédit)
    residuals = y_test - y_pred_test_final

    plt.figure(figsize=(12, 6))

    # On trace : Prix Prédit (Axe X) vs Erreur (Axe Y)
    sns.scatterplot(x=y_pred_test_final, y=residuals, alpha=0.4, color="#4C72B0")

    # Ligne rouge à 0 (là où l'erreur est nulle)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)

    plt.xlabel("Prix Prédit par le modèle (€)")
    plt.ylabel("Erreur (Réel - Prédit) (€)")
    plt.title("Analyse des Résidus : Où le modèle se trompe-t-il ?")
    plt.grid(True, alpha=0.3)

    plt.savefig(f"plots/{type}_residual_plot.png")


def regression_plot(x,y,type="xgb"):
    plt.figure(figsize=(6, 6))
    sns.regplot(x=x, y=y,  scatter_kws={"alpha": 0.5, "color": "#4C72B0"},    line_kws={"color": "#DD8452", "lw": 2}  )
    plt.xlabel("Prix réel")
    plt.ylabel("Prix prédit")
    plt.title(f"{type} – Réel vs prédit (RegPlot)")
    plt.tight_layout()
    plt.savefig(f"plots/{type}_regplot.png")
    plt.close()


def importance_plot(model, X,top_n=20):
    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

 
    # ---- Plot importance des variables ----
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=importances.head(top_n),
        x="importance",
        y="feature"
    )
    plt.title(f"XGBoost - Top {top_n} variables importantes")
    plt.xlabel("Importance")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig("plots/xgb_feature_importance.png")
    plt.close()



def regression_lineaire(df):
    features = [
        "num_rooms",
        "num_bedrooms",
        "num_bathrooms",
        "num_parking_spaces",
        "living_area_sqm",
        "total_land_area_sqm",
        "building_num_floors",
        "year_built",
        "etat_bien_num",
        "latitude",
        "longitude",
    ]

    features = [f for f in features if f in df.columns]
    print("Features utilisées :", features)

    # 2) Target
    y = df["price"]

    # On enlève les lignes où le prix est manquant
    mask = y.notna()
    y = y[mask]
    X = df.loc[mask, features]

    # 3) Imputation des valeurs manquantes (médiane pour chaque variable)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # 4) Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 5) Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 6) Modèle
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 7) Prédictions
    y_pred = model.predict(X_test)

    # 8) Metrics
    print("\n=== Performances du modèle ===")
    print(f"MAE : {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE : {root_mean_squared_error(y_test, y_pred):.2f}")
    print(f"R2 : {r2_score(y_test, y_pred):.3f}")

    # 9) Coefficients
    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": model.coef_
    }).sort_values(by="coefficient", key=np.abs, ascending=False)

    print("\n=== Importance (coefficients) ===")
    print(coef_df)

    return model, coef_df




def random_forest_regression(df):
    cols_to_remove = ["id", "dataset_source", "postal_code", "estimated_notary_fees"]
    df = df.drop(columns=cols_to_remove, errors="ignore")

    # (KMeans plante s'il y a des NaN dans lat/long)
    df_valid = df.dropna(subset=["price", "latitude", "longitude"]).copy()
    
    y = df_valid["price"]

    print("Création des clusters géographiques...")
    kmeans = KMeans(n_clusters=100, random_state=42, n_init=10)
    
    df_valid['geo_cluster'] = kmeans.fit_predict(df_valid[['latitude', 'longitude']])

    df_valid['geo_cluster'] = df_valid['geo_cluster'].astype('category')

    X = df_valid.drop(columns=["price"])
    X = X.select_dtypes(include=["number", "category"]) # On garde les nombres ET la catégorie

    print("Nombre de features utilisées par le RF :", X.shape[1])
    print("Exemples de colonnes :", list(X.columns)[:20])

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== RandomForest - Performances ===")
    print(f"MAE : {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE : {root_mean_squared_error(y_test, y_pred):.2f}")
    print(f"R2 : {r2_score(y_test, y_pred):.3f}")

    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\n=== Importance des variables (Random Forest) ===")
    print(importances.head(25))

    # ---- Plot importance des variables ----
    top_n = 20
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=importances.head(top_n),
        x="importance",
        y="feature"
    )
    plt.title(f"Random Forest - Top {top_n} variables importantes")
    plt.xlabel("Importance")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig("plots/rf_feature_importance.png")
    plt.close()


    plt.figure(figsize=(6, 6))
    sns.regplot(x=y_test, y=y_pred,   scatter_kws={"alpha": 0.5, "color": "#4C72B0"},    line_kws={"color": "#DD8452", "lw": 2}  )    
    
    plt.xlabel("Prix réel")
    plt.ylabel("Prix prédit")
    plt.title("Random Forest – Réel vs prédit (RegPlot)")
    plt.tight_layout()
    plt.savefig("plots/rf_regplot.png")
    plt.close()


    plt.close()



def xgboost_regression(df,type=None):

    if type =="A":

        df = df[df["property_type"] == 1].copy()
    elif type=="M":
        df = df[df["property_type"] == 0].copy()


    print("Type = ",type)
    print(f"Nombre d'appartements trouvés : {len(df)}")
    
    print("\n=== XGBoost - Préparation des données ===")

    cols_to_remove = ["id", "dataset_source", "postal_code", "estimated_notary_fees"]
    df = df.drop(columns=cols_to_remove, errors="ignore")

    #  Nettoyage initial
    #on va supprimer les lignes sans prix et coordonnées, je fais du feature enginneing sur les coordonées , j'ai besoin que toutes les lignes en aient
    df_valid = df.dropna(subset=["price", "latitude", "longitude"]).copy()
    y = df_valid["price"]

    # Clustering

    #le feature engineering, latitude,longitude sont pas suffisant, je crée des clusters geographiques pour obtenir des infos plus précises sur des endroits spécifiques de régions. j'utilise Kmeans pour ça
    print("Création des clusters géographiques...")
    kmeans = KMeans(n_clusters=500, random_state=42, n_init=10)
    df_valid['geo_cluster'] = kmeans.fit_predict(df_valid[['latitude', 'longitude']])
    df_valid['geo_cluster'] = df_valid['geo_cluster'].astype('category') #ici je mets en format categorie car les valeurs snt pas en echelle, la zone 1 n'est pas forcément plus riche ou plus pauvre que la zone 0, category permet a xgboost de gérere cette feature sous forme d'etiquette


    # Préparation de X
    X = df_valid.drop(columns=["price"]) # on supp le prix sinon bah pas de ML mdrr
    X = X.select_dtypes(include=["number", "category"]) # On garde que les nombres ett les catégorie (les types que xgboost supporte ) 


    X['geo_cluster'] = X['geo_cluster'].astype(int)
    
  

    # ===DECOUPAGE TRAIN/VAL/TEST 
    print("Découpage des données (Train / Validation / Test)...")
    
    # Étape A : On met de côté 15% pour le TEST FINAL (Le coffre-fort)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # Étape B : On coupe le reste (85%) en TRAIN et VALIDATION
    # 0.176 * 0.85 ≈ 0.15 (donc onn aura bien environ 15% du total en validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42
    )
    
    # Résumé des tailles
    print(f"--> Train : {len(X_train)} lignes")
    print(f"--> Val   : {len(X_val)} lignes (pour arrêter le modèle)")
    print(f"--> Test  : {len(X_test)} lignes (pour la note finale)")

    # c'est la meilleur combinaison de paramtre que j'ai trouvé, mais c'est surement pas optimal
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7, #si overfitting trop fort, à diminuer 
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1, #L1 (lasso) , augmnenter si overfitting
        reg_lambda=1.5, #L2 (puni les grosses valeurs) 
        random_state=42,
        n_jobs=-1, #workers parallèles
        tree_method="hist",
        early_stopping_rounds=50 #  permet d'arreter l'entrainement si la performance sur le jeu de validation n'augmente plus depuis 50 itérations, ça évite l'overfitting
    )


    # ok point important, on passe par le log pour eviter que les grosses valeurs de prix (biens de luxe) n'influencent trop le modèle, on va donc entrainer le modèle sur le log du prix, et à la fin on repasse en euros avec expm1 (inverse de log1p)
    # je vois une grosse diff quand je fais ça, avant le model paniquait quand y'avait des biens très chère et modifiait trop les prédictions pour les biens standards, là ça va mieux 
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val) # On transforme aussi la validation !

    # On entraîne sur TRAIN, et on surveille l'erreur sur VAL
    print("\nEntraînement en cours...")
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)], 
        verbose=False
    )

    # PRÉDICTIONS FINALES (Sur le TEST que le modèle n'a jamais vu)
    print("Calcul des scores sur le Test Set...")
    y_pred_test_log = model.predict(X_test)
    y_pred_test_final = np.expm1(y_pred_test_log) # Retour en Euros

    # Aussi sur le train pour comparer l'overfitting
    y_pred_train_log = model.predict(X_train)
    y_pred_train_final = np.expm1(y_pred_train_log)

    #  CALCUL DES MÉTRIQUES
    mae_test = mean_absolute_error(y_test, y_pred_test_final)
    rmse_test = root_mean_squared_error(y_test, y_pred_test_final)
    r2_test = r2_score(y_test, y_pred_test_final)

    r2_train = r2_score(y_train, y_pred_train_final)
    mae_train = mean_absolute_error(y_train, y_pred_train_final)
    rmse_train = root_mean_squared_error(y_train, y_pred_train_final)

    #  AFFICHAGE PROPRE
    print("\n" + "="*60)
    print(f"{'MÉTRIQUE':<10} | {'TRAIN (Apprentissage)':<20} | {'TEST (Note Finale)':<20}")
    print("="*60)
    print(f"{'R²':<10} | {r2_train:.2%}             | {r2_test:.2%}")
    print(f"{'MAE':<10} | {mae_train:,.0f} €             | {mae_test:,.0f} €")
    print(f"{'RMSE':<10} | {rmse_train:,.0f} €             | {rmse_test:,.0f} €")
    print("="*60)


    # Analyse Overfitting
    gap = r2_train - r2_test
    print(f"\nÉcart R² (Train - Test) : {gap:.2%}")
    if gap > 0.15:
        print(" ALERTE : Potentiel gros surapprentissage.")
    elif gap > 0.05:
        print("Léger surapprentissage acceptable.")
    else:
        print("Excellent : Modèle très robuste.")



    importances = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)


    # Afffichage dans le terminal
    print("\n=== Importance des variables (Top 25) ===")
    print(importances.head(25))

    # ---- Plot importance des variables ----
    importance_plot(model, X)
    # ---- Plot Réel vs Prédit ----
    regression_plot(y_test, y_pred_test_final, type="xgb")
    # ---- Plot Résidus ----
    residuals_plot(y_test, y_pred_test_final)
   
    return model





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


    #enleve les logements de luxe, limite à fixer, interessant  sur le model des maisons dans achat , (logique vu à quel point le prix peut être variable ici)
    #df = enleve_luxe(df,limite_prix=2000000)


   
    #random_forest_regression(df)

    xgboost_regression(df,"A")


if __name__ == "__main__":
    main()
