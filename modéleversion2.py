### Import des librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

#################################################
# Chargement des données
#################################################

df_conso = pd.read_csv("data_lake/releves_clean.csv")
df_meteo = pd.read_csv("donnees/meteo.csv")
df_compteurs = pd.read_csv("donnees/compteurs.csv")

#################################################
# Nettoyage des données
#################################################

# Conversion des dates
df_conso["date"] = pd.to_datetime(df_conso["date"])
df_meteo["date"] = pd.to_datetime(df_meteo["date"])

# Suppression des doublons
df_conso = df_conso.drop_duplicates()

# Suppression des valeurs négatives
df_conso = df_conso[df_conso["consommation_kwh"] >= 0]

# Gestion des valeurs manquantes
df_conso["consommation_kwh"] = (
    df_conso["consommation_kwh"]
    .interpolate()
)

df_meteo = df_meteo.ffill()

print("Valeurs manquantes consommation")
print(df_conso.isnull().sum())

print("\nValeurs manquantes météo")
print(df_meteo.isnull().sum())

#################################################
# Agrégation consommation
#################################################

df_daily = (
    df_conso
    .groupby("date")["consommation_kwh"]
    .sum()
    .reset_index()
)

#################################################
# Préparation météo
#################################################

df_meteo_daily = (
    df_meteo
    .groupby("date")
    .agg({
        "temp_moyenne_c": "mean",
        "temp_min_c": "mean",
        "temp_max_c": "mean"
    })
    .reset_index()
)

#################################################
# Fusion
#################################################

df = pd.merge(
    df_daily,
    df_meteo_daily,
    on="date",
    how="left"
)

#################################################
# Création des variables
#################################################

df["jour_semaine"] = df["date"].dt.dayofweek
df["mois"] = df["date"].dt.month

df["weekend"] = (
    df["jour_semaine"]
    .isin([5, 6])
    .astype(int)
)

# Variables historiques
df["lag_1"] = df["consommation_kwh"].shift(1)
df["lag_7"] = df["consommation_kwh"].shift(7)

# Moyenne mobile
df["rolling_7"] = (
    df["consommation_kwh"]
    .rolling(7)
    .mean()
    .shift(1)
)

# Température moyenne
df["temp"] = df["temp_moyenne_c"]

#################################################
# Suppression des lignes incomplètes
#################################################

df = df.dropna()

#################################################
# Visualisation consommation
#################################################

plt.figure(figsize=(15,5))

plt.plot(
    df["date"],
    df["consommation_kwh"]
)

plt.title("Consommation quotidienne")
plt.xlabel("Date")
plt.ylabel("Consommation (kWh)")
plt.show()

#################################################
# Variables explicatives
#################################################

features = [
    "temp",
    "jour_semaine",
    "mois",
    "weekend",
    "lag_1",
    "lag_7",
    "rolling_7"
]

target = "consommation_kwh"

#################################################
# Train / Test
#################################################

train = df[df["date"] < "2025-07-01"]
test = df[df["date"] >= "2025-07-01"]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

#################################################
# Modèles
#################################################

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    ),
    "GradientBoosting": GradientBoostingRegressor(
        random_state=42
    )
}

#################################################
# Évaluation des modèles
#################################################

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    rmse = np.sqrt(
        mean_squared_error(y_test, preds)
    )

    r2 = r2_score(y_test, preds)

    results.append([
        name,
        mae,
        rmse,
        r2
    ])

#################################################
# Résultats
#################################################

df_results = pd.DataFrame(
    results,
    columns=[
        "Modèle",
        "MAE",
        "RMSE",
        "R²"
    ]
)

print("\nRésultats des modèles :")
print(df_results)

#################################################
# Sélection du meilleur modèle
#################################################

best_model_name = (
    df_results
    .sort_values(
        by="R²",
        ascending=False
    )
    .iloc[0]["Modèle"]
)

print("\nMeilleur modèle :", best_model_name)

best_model = models[best_model_name]

best_model.fit(
    X_train,
    y_train
)

#################################################
# Prédictions finales
#################################################

predictions = best_model.predict(X_test)

#################################################
# Métriques finales
#################################################

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("\nPerformances du modèle retenu")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

#################################################
# Graphique Réel vs Prédit
#################################################

plt.figure(figsize=(15,5))

plt.plot(
    test["date"],
    y_test,
    label="Réel"
)

plt.plot(
    test["date"],
    predictions,
    label="Prédit"
)

plt.title(
    f"Prévision de consommation - {best_model_name}"
)

plt.xlabel("Date")
plt.ylabel("Consommation (kWh)")
plt.legend()

plt.show()

#################################################
# Sauvegarde du modèle
#################################################

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\nModèle sauvegardé : models/best_model.pkl")