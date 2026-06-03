#Ce script s'occupe uniquement de l'ingestion et du stockage dans le Data Lake
import pandas as pd
import os

os.makedirs("data_lake", exist_ok=True)

# Chargement
df_conso = pd.read_csv(
    "donnees/releves_consommation.csv"
)

# Nettoyage
df_conso = df_conso.drop_duplicates()

df_conso = df_conso[
    df_conso["consommation_kwh"] >= 0
]

# Sauvegarde
df_conso.to_csv(
    "data_lake/releves_clean.csv",
    index=False
)

print("Pipeline terminé")