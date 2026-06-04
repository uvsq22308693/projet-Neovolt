import pandas as pd

df = pd.read_csv(r"C:\Users\dell\Downloads\donnees\donnees_propres\clean_releves_consommation.csv")

df["segment_consommation"] = pd.cut(
    df.groupby("id_pdl")["consommation_kwh"].transform("mean"),
    bins=[0, 15, 50, float("inf")],
    labels=["faible", "moyen", "fort"]
)

print(df["segment_consommation"].value_counts())