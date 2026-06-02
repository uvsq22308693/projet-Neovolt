import pandas as pd

df = pd.read_csv(r"C:\Users\dell\Downloads\donnees\donnees_propres\clean_releves_consommation.csv")

df_consommation_client = df.groupby("id_pdl")["consommation_kwh"].sum()

top_10_clients = df_consommation_client.sort_values(ascending=False).head(10)
print("\n TOP 10 CLIENTS LES PLUS CONSOMMATEURS :\n")
print(top_10_clients)