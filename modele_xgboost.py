import os
import pickle
import numpy as np
import xgboost as xgb

# Récupérer le chemin absolu du répertoire courant
current_dir = os.path.dirname(os.path.abspath(__file__))

# Charger le modèle XGBoost
model_path = os.path.join(current_dir, 'mon_modele.json')
print(f"Chargement du modèle depuis : {model_path}")
model = xgb.Booster(model_file=model_path)

# Charger le seuil de décision
threshold_path = os.path.join(current_dir, 'seuil_decision.pkl')
print(f"Chargement du seuil de décision depuis : {threshold_path}")
with open(threshold_path, 'rb') as file:
    threshold = pickle.load(file)
    print(f"Seuil de décision chargé : {threshold}")

# Fonction pour faire des prédictions avec le modèle chargé
def predict(X):
    print("Données d'entrée X :")
    print(X)
    print(f"Nombre d'instances : {len(X)}")
    print(f"Nombre de colonnes : {len(X[0])}")
    print("Types de données des colonnes :")
    for i, instance in enumerate(X[:5]):  # Affiche les 5 premières instances
        print(f"Instance {i} : {[type(val) for val in instance]}")

    dmatrix = xgb.DMatrix(np.array(X), missing=np.inf)
    prediction_proba = model.predict(dmatrix)
    print("Types de données des probabilités de prédiction :")
    print(type(prediction_proba))
    if isinstance(prediction_proba, np.ndarray):
        prediction_proba = prediction_proba.tolist()
        print("Conversion des probabilités de prédiction en liste Python")
    else:
        print("Les probabilités de prédiction sont déjà une liste Python")

    prediction = [1 if proba >= threshold else 0 for proba in prediction_proba]
    print("Types de données des prédictions :")
    print(type(prediction))
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
        print("Conversion des prédictions en liste Python")
    else:
        print("Les prédictions sont déjà une liste Python")

    return prediction, prediction_proba
