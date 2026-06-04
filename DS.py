###Import des librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

###vérifier qu’il n’y a plus de données manquantes
print(df_conso.isnull().sum())
print(df_meteo.isnull().sum())

###Agrégation de la consommation (j'ai regroupé les consommations par date afin d'obtenir une consommation quotidienne totale.)
df_daily = (
    df_conso
    .groupby("date")["consommation_kwh"]
    .sum()
    .reset_index()
)
####Préparation météo (groupées par jour afin d'obtenir une température moyenne quotidienne.)
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
###Fusion des données, jeu de données unique pour l'apprentissage.

df = pd.merge(df_daily, df_meteo_daily, on="date", how="left")

###Création des variables Feature Engineering
# Variables temporelles utiles
df["jour_semaine"] = df["date"].dt.dayofweek
df["mois"] = df["date"].dt.month

# Week-end 
df["weekend"] = (df["jour_semaine"].isin([5,6])).astype(int)

# Cette variable représente la consommation de la veille
df["lag_1"] = df["consommation_kwh"].shift(1)
# Cette variable représente la consommation observée une semaine auparavant.
df["lag_7"] = df["consommation_kwh"].shift(7)

# Cette variable représente la moyenne des consommations des sept derniers jours
df["rolling_7"] = df["consommation_kwh"].rolling(7).mean().shift(1)

# Météo simplifiée
df["temp"] = df["temp_moyenne_c"]


###Suppression des lignes incomplètes
df = df.dropna()

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

###Prédictions (Le modèle essaie de deviner la consommation sur les données jamais vues)
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


####Comparaison des modèles
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


# Création du dossier models
os.makedirs("models", exist_ok=True)

# Ré-entraînement du meilleur modèle (Gradient Boosting)
best_model = GradientBoostingRegressor(random_state=42)
best_model.fit(X_train, y_train)

# Sauvegarde du modèle final
joblib.dump(best_model, "models/gradient_boosting_v1.pkl")

