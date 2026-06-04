import pandas as pd

df_cons = pd.read_csv(r"C:\Users\dell\Downloads\donnees\donnees_propres\clean_releves_consommation.csv")
df_compteurs = pd.read_csv(r"C:\Users\dell\Downloads\donnees\donnees_propres\clean_compteurs.csv")
df_clients = pd.read_csv(r"C:\Users\dell\Downloads\donnees\donnees_propres\clean_clients.csv")
df_meteo = pd.read_csv(r"C:\Users\dell\Downloads\donnees\donnees_propres\clean_meteo.csv")

# 🌡️ météo
df_cons["date"] = pd.to_datetime(df_cons["date"])
df_meteo["date"] = pd.to_datetime(df_meteo["date"])

df_merge = df_cons.merge(df_meteo, on=["date", "zone"])

print("\n🌡️ Corrélation température vs consommation :")
print(df_merge[["consommation_kwh", "temp_moyenne_c"]].corr())

# 👥 clients
df_full = df_cons.merge(df_compteurs, on="id_pdl")
df_full = df_full.merge(df_clients, on="id_client")

# 🔥 nettoyage noms colonnes zone
df_full = df_full.rename(columns={"zone_x": "zone"})

# 🌍 zone vs consommation
print("\n🌍 Zone vs consommation :")
print(df_full.groupby("zone")["consommation_kwh"].mean())

# 👥 type client
print("\n👥 Type client vs consommation :")
print(df_full.groupby("type_client")["consommation_kwh"].mean())