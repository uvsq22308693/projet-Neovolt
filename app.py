from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/best_model.pkl")

@app.get("/predict")
def predict(
    temp: float,
    jour_semaine: int,
    mois: int,
    weekend: int,
    lag_1: float,
    lag_7: float,
    rolling_7: float
):

    data = pd.DataFrame([{
        "temp": temp,
        "jour_semaine": jour_semaine,
        "mois": mois,
        "weekend": weekend,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "rolling_7": rolling_7
    }])

    prediction = model.predict(data)

    return {
        "prediction_kwh": float(prediction[0])
    }