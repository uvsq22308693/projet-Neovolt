from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="API Détection de Fraude Neovolt",
    version="1.0"
)

# Chargement du modèle
model = joblib.load("../modele_fraude.pkl")


class ClientData(BaseModel):
    consommation_kwh: float
    conso_j_1: float
    moyenne_7j: float
    std_7j: float
    variation_pct: float
    conso_par_m2: float
    jour_semaine: int
    mois: int
    weekend: int
    surface_m2: float
    nb_personnes_foyer: int
    puissance_souscrite_kva: float


@app.get("/")
def home():
    return {"message": "API Détection de Fraude Neovolt"}


@app.post("/predict")
def predict(data: ClientData):

    df = pd.DataFrame([data.dict()])

    prediction = int(model.predict(df)[0])

    probability = float(
        model.predict_proba(df)[0][1]
    )

    return {
        "prediction": prediction,
        "probabilite_fraude": round(probability, 4)
    }