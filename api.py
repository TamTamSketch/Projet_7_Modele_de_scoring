import flask
from flask_debugtoolbar import DebugToolbarExtension
import importlib.metadata
import numpy as np
import os
import sys
import xgboost as xgb
import logging
from logging.handlers import RotatingFileHandler


# Récupérer le chemin absolu du répertoire courant
current_dir = os.path.dirname(os.path.abspath(__file__))
# Ajouter le répertoire courant au chemin de recherche de Python
sys.path.append(current_dir)
# Importer le module modele_xgboost depuis le répertoire courant
from modele_xgboost import predict


# Configurer les logs
file_handler = RotatingFileHandler('flask_app.log', maxBytes=16384, backupCount=20)
file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.INFO)

app = flask.Flask(__name__)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

flask_version = importlib.metadata.version('flask')
print(f"Version de Flask : {flask_version}")

@app.route('/')
def index():
    return 'Bienvenue sur mon API Flask !'

@app.route('/predict', methods=['GET', 'POST'])
def predict_endpoint():
    app.logger.info(f"Requête reçue à l'URL : {flask.request.url}")
    data = flask.request.get_json()['data']
    X = np.array([np.array(instance) for instance in data])

    try:         
        prediction, prediction_proba = predict(X)
        app.logger.info('Prédiction effectuée avec succès')
        return flask.jsonify({
            'prediction': prediction,
            'prediction_proba': prediction_proba
        })
    except Exception as e:
        app.logger.error(f'Erreur lors de la prédiction : {e}')
        return flask.jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Démarrage du serveur Flask..")
    app.run(host='0.0.0.0',port=5000,debug=True)
    print("Serveur Flask démarré")