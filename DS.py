###Import des librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

results = []

###Chargement des données
df_conso = pd.read_csv("donnees/releves_consommation.csv")
df_meteo = pd.read_csv("donnees/meteo.csv")
df_compteurs = pd.read_csv("donnees/compteurs.csv")

###Nettoyage des données
###Conversion des dates
df_conso["date"] = pd.to_datetime(df_conso["date"])
df_meteo["date"] = pd.to_datetime(df_meteo["date"])

###Suppression des doublons
df_conso = df_conso.drop_duplicates()

###Suppression des consommations négatives
df_conso = df_conso[df_conso["consommation_kwh"] >= 0]

###Gestion des valeurs manquantes
df_conso["consommation_kwh"] = (
    df_conso["consommation_kwh"]
    .interpolate()
)

df_meteo = df_meteo.ffill()

print(df_conso.isnull().sum())
print(df_meteo.isnull().sum())

###Agrégation de la consommation
df_daily = (
    df_conso
    .groupby("date")["consommation_kwh"]
    .sum()
    .reset_index()
)
####Préparation météo
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
###Fusion des données

df = pd.merge(df_daily, df_meteo_daily, on="date", how="left")

###Création des variables
# Variables temporelles utiles
df["jour_semaine"] = df["date"].dt.dayofweek
df["mois"] = df["date"].dt.month

# Week-end (Bon signal)
df["weekend"] = (df["jour_semaine"].isin([5,6])).astype(int)

# Lag (garder uniquement essentiels)
df["lag_1"] = df["consommation_kwh"].shift(1)
df["lag_7"] = df["consommation_kwh"].shift(7)

# Rolling (garder 1 seule)
df["rolling_7"] = df["consommation_kwh"].rolling(7).mean().shift(1)

# Météo simplifiée
df["temp"] = df["temp_moyenne_c"]


###Suppression des lignes incomplètes
df = df.dropna()
###Visualisation
plt.figure(figsize=(15,5))

plt.plot(
    df["date"],
    df["consommation_kwh"]
)

plt.title("Consommation quotidienne")
plt.xlabel("Date")
plt.ylabel("kWh")

plt.show()
###Définition des variables
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

###Séparation Train / Test
train = df[df["date"] < "2025-07-01"]
test = df[df["date"] >= "2025-07-01"]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

###Entraînement du modèle
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

###Prédictions
predictions = model.predict(X_test)

###Évaluation

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

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append([name, mae, rmse, r2])


df_results = pd.DataFrame(results, columns=["Modèle", "MAE", "RMSE", "R²"])
print(df_results)

###Visualisation des résultats
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

plt.title("Prévision de consommation")
plt.xlabel("Date")
plt.ylabel("kWh")

plt.legend()

plt.show()


import joblib

joblib.dump(model, "models/gradient_boosting_v1.pkl")