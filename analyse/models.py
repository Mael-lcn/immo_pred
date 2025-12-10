import argparse

import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from outils import load_all_regions
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


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
    os.makedirs("plots", exist_ok=True)
    cols_to_remove = ["id", "dataset_source", "postal_code", "estimated_notary_fees"]
    df = df.drop(columns=cols_to_remove, errors="ignore")

    df_valid = df[df["price"].notna()].copy()
    y = df_valid["price"]

    # Si le OHE est déjà fait, toutes les colonnes utiles sont numériques
    X = df_valid.drop(columns=["price"])

    # On garde seulement les numériques (au cas où il reste des colonnes non num)
    X = X.select_dtypes(include="number")

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



def xgboost_regression(df):
    os.makedirs("plots", exist_ok=True)

    print("\n=== XGBoost - Préparation des données ===")

    cols_to_remove = ["id", "dataset_source", "postal_code", "estimated_notary_fees"]
    df = df.drop(columns=cols_to_remove, errors="ignore")

    df_valid = df[df["price"].notna()].copy()
    y = df_valid["price"]

    # On enlève juste la target
    X = df_valid.drop(columns=["price"])
    X = X.select_dtypes(include="number")

    print("\nColonnes XGBoost :")
    print(list(X.columns))
    print("Nombre de features utilisées par XGBoost :", X.shape[1])

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=0.2, random_state=42
    )

    print("\n=== Entraînement du modèle XGBoost ===")

    model = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== XGBoost - Performances ===")
    print(f"MAE : {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE : {root_mean_squared_error(y_test, y_pred):.2f}")
    print(f"R2 : {r2_score(y_test, y_pred):.3f}")

    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\n=== Importance des variables (XGBoost) ===")
    print(importances.head(25))

    # ---- Plot importance des variables ----
    top_n = 20
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


    plt.figure(figsize=(6, 6))
    sns.regplot(x=y_test, y=y_pred,  scatter_kws={"alpha": 0.5, "color": "#4C72B0"},    line_kws={"color": "#DD8452", "lw": 2}  )
    plt.xlabel("Prix réel")
    plt.ylabel("Prix prédit")
    plt.title("XGBoost – Réel vs prédit (RegPlot)")
    plt.tight_layout()
    plt.savefig("plots/xgb_regplot.png")
    plt.close()


    plt.close()



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


    #regression_lineaire(df)
    random_forest_regression(df)
    xgboost_regression(df)


if __name__ == "__main__":
    main()
