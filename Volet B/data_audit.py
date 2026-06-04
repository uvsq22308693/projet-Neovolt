import pandas as pd
import os

# 📁 dossier des fichiers CSV
dossier = r"C:\Users\dell\Downloads\donnees\donnees"

# 📊 liste pour stocker les résultats
resultats = []

# 🔁 parcourir les fichiers
for fichier in os.listdir(dossier):
    if fichier.endswith(".csv"):
        chemin = os.path.join(dossier, fichier)
        
        try:
            df = pd.read_csv(chemin)
            
            resultats.append({
                "table": fichier,
                "nb_lignes": df.shape[0],
                "valeurs_manquantes": df.isnull().sum().sum()
            })
        
        except Exception as e:
            resultats.append({
                "table": fichier,
                "nb_lignes": "Erreur",
                "valeurs_manquantes": str(e)
            })

# 📊 transformer en tableau pandas
df_resultat = pd.DataFrame(resultats)

# 🖥️ afficher le tableau
print(df_resultat)