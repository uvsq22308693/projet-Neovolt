import numpy as np
import pandas as pd

def preprocess(consommation, compteurs, clients, fraudes):

    # =====================
    # Conversion des dates
    # =====================
    consommation["date"] = pd.to_datetime(consommation["date"], errors="coerce")
    clients["date_entree"] = pd.to_datetime(clients["date_entree"], errors="coerce")
    compteurs["date_pose"] = pd.to_datetime(compteurs["date_pose"], errors="coerce")
    fraudes["date_detection"] = pd.to_datetime(fraudes["date_detection"], errors="coerce")

    # =====================
    # Nettoyage des données
    # =====================
    consommation = consommation.drop_duplicates()

    consommation["consommation_kwh"] = pd.to_numeric(
        consommation["consommation_kwh"], errors="coerce"
    )

    consommation = consommation[
        consommation["consommation_kwh"].isna()
        | (consommation["consommation_kwh"] >= 0)
    ]

    consommation["consommation_kwh"] = consommation["consommation_kwh"].interpolate()
    consommation = consommation.dropna(subset=["consommation_kwh"])

    clients["nb_personnes_foyer"] = clients["nb_personnes_foyer"].fillna(0)
    clients["surface_m2"] = clients["surface_m2"].fillna(clients["surface_m2"].median())

    compteurs = compteurs[compteurs["statut"] == "actif"]

    if "zone" in compteurs.columns:
        compteurs = compteurs.drop(columns=["zone"])

    # =====================
    # Fusion des données
    # =====================
    df = consommation.merge(compteurs, on="id_pdl", how="left")
    df = df.merge(clients, on="id_client", how="left")

    print(df.shape)

    # =====================
    # Feature Engineering
    # =====================
    df = df.sort_values(["id_pdl", "date"])

    df["conso_j_1"] = df.groupby("id_pdl")["consommation_kwh"].shift(1)

    df["moyenne_7j"] = df.groupby("id_pdl")["consommation_kwh"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    df["std_7j"] = df.groupby("id_pdl")["consommation_kwh"].transform(
        lambda x: x.rolling(7, min_periods=1).std()
    )

    df["variation_pct"] = df.groupby("id_pdl")["consommation_kwh"].pct_change()

    df["conso_par_m2"] = df["consommation_kwh"] / df["surface_m2"].replace(0, np.nan)

    df["jour_semaine"] = df["date"].dt.dayofweek
    df["mois"] = df["date"].dt.month
    df["weekend"] = (df["jour_semaine"] >= 5).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # =====================
    # Label fraude
    # =====================
    fraudes_ids = set(fraudes["id_pdl"].unique())
    df["fraude"] = df["id_pdl"].isin(fraudes_ids).astype(int)

    print(df["fraude"].value_counts())

    # =====================
    # Baseline anomaly detection
    # =====================
    df["baseline_anomalie"] = np.where(
        df["consommation_kwh"]
        > (df["moyenne_7j"] + 3 * df["std_7j"].fillna(0)),
        1,
        0
    )

    print(df["baseline_anomalie"].sum())

    return df