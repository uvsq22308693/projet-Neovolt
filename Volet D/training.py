from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import joblib

FEATURES = [
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

def train(df):

    # =====================
    # Dataset ML
    # =====================
    X = df[FEATURES].replace([float("inf"), float("-inf")], 0).fillna(0).astype(float)
    y = df["fraude"]

    # =====================
    # Equilibrage dataset
    # =====================
    df_majority = df[df["fraude"] == 0]
    df_minority = df[df["fraude"] == 1]

    df_majority = df_majority.sample(n=len(df_minority), random_state=42)
    df_balanced = pd.concat([df_majority, df_minority]).sample(frac=1)

    X = df_balanced[FEATURES].fillna(0)
    y = df_balanced["fraude"]

    # =====================
    # Split
    # =====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # =====================
    # Models
    # =====================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = {}
    best_model = None
    best_f1 = 0

    for name, model in models.items():
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, pred, output_dict=True)
        auc = roc_auc_score(y_test, proba)

        f1 = report["1"]["f1-score"]

        results[name] = {"f1": f1, "auc": auc}

        if f1 > best_f1:
            best_model = model
            best_f1 = f1

    # =====================
    # Save model
    # =====================
    joblib.dump(best_model, "modele_fraude.pkl")

    print("Modèle sauvegardé ✔")

    return best_model, results