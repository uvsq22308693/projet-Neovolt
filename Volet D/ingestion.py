import pandas as pd
from pathlib import Path

def load_data(data_path="donnees"):
    DATA_PATH = Path(data_path)

    consommation = pd.read_csv(DATA_PATH / "releves_consommation.csv")
    compteurs = pd.read_csv(DATA_PATH / "compteurs.csv")
    clients = pd.read_csv(DATA_PATH / "clients.csv")
    fraudes = pd.read_csv(DATA_PATH / "cas_fraude_confirmes.csv")

    print("Fichiers chargés")

    return consommation, compteurs, clients, fraudes