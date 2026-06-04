import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib


def load_data(data_path="donnees"):
    data_path = Path(data_path)

    consommation = pd.read_csv(data_path / "releves_consommation.csv")
    compteurs = pd.read_csv(data_path / "compteurs.csv")
    clients = pd.read_csv(data_path / "clients.csv")
    fraudes = pd.read_csv(data_path / "cas_fraude_confirmes.csv")

    print("Fichiers chargés")
    return consommation, compteurs, clients, fraudes


def preprocess(consommation, compteurs, clients, fraudes):
    # dates
    consommation["date"] = pd.to_datetime(consommation["date"], errors="coerce")
    clients["date_entree"] = pd.to_datetime(clients["date_entree"], errors="coerce")
    compteurs["date_pose"] = pd.to_datetime(compteurs["date_pose"], errors="coerce")
    fraudes["date_detection"] = pd.to_datetime(fraudes["date_detection"], errors="coerce")

    # nettoyage consommation
    consommation = consommation.drop_duplicates()
    consommation["consommation_kwh"] = pd.to_numeric(consommation["consommation_kwh"], errors="coerce")

    consommation = consommation[
        consommation["consommation_kwh"].isna()
        | (consommation["consommation_kwh"] >= 0)
    ]

    consommation["consommation_kwh"] = consommation["consommation_kwh"].interpolate()
    consommation = consommation.dropna(subset=["consommation_kwh"])

    # clients
    clients["nb_personnes_foyer"] = clients["nb_personnes_foyer"].fillna(0)
    clients["surface_m2"] = clients["surface_m2"].fillna(clients["surface_m2"].median())

    # compteurs
    compteurs = compteurs[compteurs["statut"] == "actif"]
    if "zone" in compteurs.columns:
        compteurs = compteurs.drop(columns=["zone"])

    # merge
    df = consommation.merge(compteurs, on="id_pdl", how="left")
    df = df.merge(clients, on="id_client", how="left")

    print("Shape après fusion:", df.shape)

    return df, fraudes


def feature_engineering(df, fraudes):
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

    # label fraude
    fraudes_ids = set(fraudes["id_pdl"].unique())
    df["fraude"] = df["id_pdl"].isin(fraudes_ids).astype(int)

    print("Distribution fraude:\n", df["fraude"].value_counts())

    # baseline anomalie
    df["baseline_anomalie"] = np.where(
        df["consommation_kwh"] > (df["moyenne_7j"] + 3 * df["std_7j"].fillna(0)),
        1,
        0
    )

    return df


def balance_dataset(df, features):
    df_majority = df[df["fraude"] == 0]
    df_minority = df[df["fraude"] == 1]

    n_minority = len(df_minority)
    df_majority_downsampled = df_majority.sample(n=n_minority, random_state=42)

    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Distribution après équilibrage:\n", df_balanced["fraude"].value_counts())

    X = df_balanced[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = df_balanced["fraude"]

    return X, y


def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=300, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = []

    for name, model in models.items():
        print("\n---", name, "---")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        auc = roc_auc_score(y_test, y_proba)
        print("AUC:", auc)

        report = classification_report(y_test, y_pred, output_dict=True)

        results.append({
            "modele": name,
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"],
            "auc": auc
        })

    return models, pd.DataFrame(results)


def main():
    consommation, compteurs, clients, fraudes = load_data()

    df, fraudes = preprocess(consommation, compteurs, clients, fraudes)

    df = feature_engineering(df, fraudes)

    features = [
        "consommation_kwh",
        "conso_j_1",
        "moyenne_7j",
        "std_7j",
        "variation_pct",
        "conso_par_m2",
        "jour_semaine",
        "mois",
        "weekend",
        "surface_m2",
        "nb_personnes_foyer",
        "puissance_souscrite_kva"
    ]

    X, y = balance_dataset(df, features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models, results_df = train_models(X_train, X_test, y_train, y_test)

    print("\n--- Résultats finaux ---")
    print(results_df.sort_values("f1", ascending=False))

    best_model = models["Gradient Boosting"]
    joblib.dump(best_model, "modele_fraude.pkl")
    print("Modèle sauvegardé ✔")

    # test inference
    model = joblib.load("modele_fraude.pkl")

    new_data = pd.DataFrame([{
        "consommation_kwh": 120,
        "conso_j_1": 110,
        "moyenne_7j": 100,
        "std_7j": 15,
        "variation_pct": 0.18,
        "conso_par_m2": 2.5,
        "jour_semaine": 2,
        "mois": 6,
        "weekend": 0,
        "surface_m2": 50,
        "nb_personnes_foyer": 3,
        "puissance_souscrite_kva": 6
    }])

    pred = model.predict(new_data)[0]
    print("\nPrédiction :", pred)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(new_data)[0][1]
        print("Probabilité fraude :", proba)


if __name__ == "__main__":
    main()