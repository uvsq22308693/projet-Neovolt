import pandas as pd
import matplotlib.pyplot as plt

# 📁 charger les données (version nettoyée recommandée)
df = pd.read_csv(r"C:\Users\dell\Downloads\donnees\donnees_propres\clean_releves_consommation.csv")

# =========================
# 1️⃣ CONVERSION DATE
# =========================
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# =========================
# 2️⃣ SUPPRESSION DATES INVALIDES (sécurité)
# =========================
df = df.dropna(subset=["date"])

# =========================
# 3️⃣ EXTRACTION MOIS
# =========================
df["mois"] = df["date"].dt.month

# =========================
# 4️⃣ DÉFINITION DES SAISONS
# =========================
def saison(mois):
    if mois in [12, 1, 2]:
        return "hiver"
    elif mois in [3, 4, 5]:
        return "printemps"
    elif mois in [6, 7, 8]:
        return "été"
    else:
        return "automne"

df["saison"] = df["mois"].apply(saison)

# =========================
# 5️⃣ ANALYSE SAISONNIÈRE
# =========================

# 📊 consommation moyenne par mois
cons_mois = df.groupby("mois")["consommation_kwh"].mean()

# 📊 consommation moyenne par saison
cons_saison = df.groupby("saison")["consommation_kwh"].mean()

print("\n📊 CONSOMMATION MOYENNE PAR MOIS :\n")
print(cons_mois)

print("\n📊 CONSOMMATION MOYENNE PAR SAISON :\n")
print(cons_saison)

# =========================
# 6️⃣ VISUALISATION
# =========================

plt.figure()

cons_mois.plot(marker="o")
plt.title("Saisonnalité de la consommation énergétique (par mois)")
plt.xlabel("Mois")
plt.ylabel("Consommation moyenne (kWh)")
plt.grid()
plt.show()