import pytest
import json
from api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

#Vérifie que la route racine / retourne le bon message de bienvenue.
def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.data == b'Bienvenue sur mon API Flask !'

#Vérifie que l'endpoint /predict retourne une prédiction et une probabilité de prédiction correctement lorsqu'on lui envoie des données valides.
def test_predict_endpoint_success(client):
    data = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10]
    ]
    response = client.post('/predict', json={'data': data})
    assert response.status_code == 200
    assert 'prediction' in json.loads(response.data)
    assert 'prediction_proba' in json.loads(response.data)

#Vérifie que l'endpoint /predict retourne une erreur avec un code 500 lorsqu'on lui envoie des données invalides.
def test_predict_endpoint_error(client):
    data = [
        ['a', 'b', 'c', 'd', 'e']
    ]
    response = client.post('/predict', json={'data': data})
    assert response.status_code == 500
    assert 'error' in json.loads(response.data)