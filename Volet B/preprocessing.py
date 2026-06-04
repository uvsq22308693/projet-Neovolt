import pandas as pd
import os

# dossier source
dossier_source = r"C:\Users\dell\Downloads\donnees\donnees"

# dossier de sortie
dossier_sortie = r"C:\Users\dell\Downloads\donnees\donnees_propres"
os.makedirs(dossier_sortie, exist_ok=True)

# 🔁 parcourir les fichiers CSV
for fichier in os.listdir(dossier_source):
    if fichier.endswith(".csv"):
        chemin = os.path.join(dossier_source, fichier)

        try:
            df = pd.read_csv(chemin)

            # =========================
            # SUPPRESSION DOUBLONS
            # =========================
            df = df.drop_duplicates()

            # suppression des lignes avec NaN (simple et efficace pour audit)
            df = df.dropna()

            # =========================
            # CORRECTION FORMATS DATES
            # =========================
            for col in df.columns:
                if "date" in col.lower() or "horod" in col.lower():
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            # supprimer lignes invalides après conversion dates
            df = df.dropna()

            # =========================
            # FILTRAGE VALEURS ABERRANTES
            # =========================
            for col in df.select_dtypes(include=["float64", "int64"]).columns:
                mean = df[col].mean()
                std = df[col].std()

                df = df[(df[col] >= mean - 3 * std) & (df[col] <= mean + 3 * std)]

            # =========================
            # SAUVEGARDE
            # =========================
            nouveau_nom = "clean_" + fichier
            chemin_sortie = os.path.join(dossier_sortie, nouveau_nom)

            df.to_csv(chemin_sortie, index=False)

            print(f"{fichier} nettoyé → {nouveau_nom} | lignes finales: {df.shape[0]}")

        except Exception as e:
            print(f"Erreur avec {fichier} : {e}")