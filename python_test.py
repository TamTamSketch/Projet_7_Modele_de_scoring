import json
import pandas as pd
import numpy as np
import requests

# Charger les données de test depuis un fichier CSV
df_final_test = pd.read_csv('/home/tamara-daniel-tricot/Bureau/0_Projets_ParcoursDataScientist/P7/Projet_7/df_final_test.csv')

# Vérifier les valeurs infinies
has_inf = np.isinf(df_final_test).values.any()

if has_inf:
    print("Le DataFrame contient des valeurs infinies.")
    
    # Remplacer les valeurs infinies par NaN
    df_final_test = df_final_test.replace([np.inf, -np.inf], np.nan)
    print("Valeurs infinies remplacées par NaN")
    
    # Remplacer les valeurs NaN par la moyenne de la colonne
    df_final_test = df_final_test.fillna(df_final_test.mean())
    print("Valeurs NaN remplacées par la moyenne de la colonne")

# Convertir le DataFrame en un tableau NumPy
X_test = df_final_test.values

# Convertir le tableau NumPy en une liste Python
X_test_list = X_test.tolist()

# Définir les en-têtes pour la requête POST
headers = {'Content-Type': 'application/json'}

# Convertir les données en format JSON
data = json.dumps({"data": X_test_list})

# Envoyer une requête POST à l'API Flask
response = requests.post('http://localhost:5000/predict', data=data, headers=headers)

# Vérifier le code de statut de la réponse
if response.status_code == 200:
    # Récupérer les prédictions et les probabilités à partir de la réponse de l'API
    predictions = response.json()
    print("Type de données des prédictions reçues :")
    print(type(predictions['prediction']))
    print("Type de données des probabilités de prédiction reçues :")
    print(type(predictions['prediction_proba']))
    print("Prédictions :", predictions['prediction'])
    print("Probabilités :", predictions['prediction_proba'])
else:
    # Afficher un message d'erreur si la requête a échoué
    print(f"Erreur : {response.status_code}")