import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\dell\Downloads\donnees\donnees_propres\clean_releves_consommation.csv")

df["consommation_kwh"] = df["consommation_kwh"].where(df["consommation_kwh"] >= 0, np.nan)
df = df.dropna(subset=["consommation_kwh"])

df.to_csv(r"C:\Users\dell\Downloads\donnees\donnees_propres\clean_releves_consommation.csv", index=False)