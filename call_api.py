import pandas as pd
import requests
import numpy as np

# Charger les données de test depuis un fichier CSV
df_final_test = pd.read_csv('/home/tamara-daniel-tricot/Bureau/0_Projets_ParcoursDataScientist/P7/Projet_7/df_final_test.csv')

# Prendre les 5 premières lignes de df_final_test
df_final_test_sample = df_final_test.head(100000)

# Remplacer les valeurs NaN et infinies par une valeur valide (par exemple, 0)
df_final_test_sample = df_final_test_sample.replace([np.inf, -np.inf, np.nan], 0)

# Convertir le DataFrame en un tableau NumPy
X_test_sample = df_final_test_sample.values

# Créer un dictionnaire avec la clé "data" et la valeur étant le tableau NumPy
data = {"data": X_test_sample.tolist()}

# Envoyer une requête POST à l'API Flask
response = requests.post('http://localhost:5000/predict', json=data)

# Récupérer les prédictions et les probabilités à partir de la réponse de l'API
if response.status_code == 200:
    predictions = response.json()
    print("Prédictions :", predictions['prediction'])
    print("Probabilités :", predictions['prediction_proba'])
else:
    print(f"Erreur : {response.status_code}")