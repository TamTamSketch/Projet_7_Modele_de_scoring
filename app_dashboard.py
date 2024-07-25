import numpy as np
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import shap
import xgboost as xgb
import json
import io
import base64
import plotly.graph_objs as go
import plotly.figure_factory as ff

def convert_to_serializable(obj):
    if isinstance(obj, (bool, np.bool_)):
        return int(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def send_request(features):
    url = 'http://0.0.0.0:5000/predict'
    instance_data = [convert_to_serializable(x) for x in features['instance_data']]
    data = {'data': [instance_data]}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erreur : {response.status_code}")
        return None

def afficher_score(score, probabilite_prediction):
    st.subheader("Score du client")
    
    # Créer la jauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Score de risque"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.5], 'color': "green"},
                {'range': [0.5, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 0.5
            }
        }
    ))
    
    st.plotly_chart(fig)
    
    if isinstance(probabilite_prediction, (list, tuple)) and len(probabilite_prediction) >= 2:
        couleur = "green" if score < 0.5 else "red"
        interpretation = f"Le client n'est pas à risque pour emprunter. (Probabilité : {probabilite_prediction[0]:.4f})" if score < 0.5 else f"Le client est à risque pour emprunter. (Probabilité : {probabilite_prediction[1]:.4f})"

def afficher_infos_client(id_client, df_final_test_limited):
    client = df_final_test_limited.iloc[id_client]
    st.subheader("Informations descriptives")

    # Obtenir la liste des colonnes du DataFrame
    colonnes = df_final_test_limited.columns.tolist()

    # Permettre à l'utilisateur de sélectionner les colonnes à afficher
    colonnes_selectionnees = st.multiselect("Sélectionnez les informations à afficher", colonnes)

    # Afficher les informations sélectionnées
    if colonnes_selectionnees:
        st.write(client[colonnes_selectionnees])
    else:
        st.write("Aucune information sélectionnée.")

    # Retourner les colonnes sélectionnées
    return colonnes_selectionnees

def afficher_caracteristiques_importantes(instance_data, df_final_test_limited, model):
    st.subheader("Caractéristiques les plus impactantes")

    # Créer l'explainer
    explainer = shap.TreeExplainer(model)

    # Convertir instance_data en une matrice 2D
    instance_2d = np.array(instance_data).reshape(1, -1)

    # Calculer les valeurs SHAP pour l'instance
    shap_values_instance = explainer.shap_values(instance_2d)

    # Récupérer les noms des caractéristiques
    feature_names = df_final_test_limited.columns.tolist()

    # Créer un objet d'explication SHAP
    shap_values_instance = shap_values_instance[0]  # Prendre la première (et unique) ligne
    shap_explanation = shap.force_plot(
        explainer.expected_value,
        shap_values_instance,
        instance_2d,
        feature_names=feature_names,
        plot_cmap=["#FF0000", "#0000FF"],
        matplotlib=True,
    )

    # Afficher le graphique SHAP
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(shap_explanation)

    # Afficher les 10 valeurs SHAP les plus impactantes pour l'instance
    shap_values_sorted = sorted(zip(feature_names, shap_values_instance), key=lambda x: abs(x[1]), reverse=True)
    st.write("Les 10 caractéristiques les plus impactantes pour l'instance :")
    for feature, shap_value in shap_values_sorted[:10]:
        st.write(f"{feature}: {shap_value:.3f}")

def comparer_infos(id_client, df_final_test_limited, groupe_similaire, caracteristiques_selectionnees):
    client = df_final_test_limited.iloc[id_client]
    if groupe_similaire:
        groupe = df_final_test_limited[(df_final_test_limited['EXT_SOURCE_1'] == client['EXT_SOURCE_1']) &
                                       (df_final_test_limited['EXT_SOURCE_2'] == client['EXT_SOURCE_2']) &
                                       (df_final_test_limited['EXT_SOURCE_3'] == client['EXT_SOURCE_3']) &
                                       (df_final_test_limited['CODE_GENDER'] == client['CODE_GENDER'])]
    else:
        groupe = df_final_test_limited

    fig = go.Figure()

    # Ajouter les traces pour les caractéristiques sélectionnées
    for caracteristique in caracteristiques_selectionnees:
        fig.add_trace(go.Bar(x=['Client'], y=[client[caracteristique]], name=caracteristique))
        fig.add_trace(go.Bar(x=['Groupe'], y=[groupe[caracteristique].mean()], name=f"{caracteristique} Groupe"))

    fig.update_layout(barmode='group')
    st.plotly_chart(fig)

    # Afficher le genre majoritaire pour le groupe
    code_gender_counts = groupe['CODE_GENDER'].value_counts()
    genre_majoritaire_code = code_gender_counts.idxmax()
    st.write(f"Le genre majoritaire pour le groupe est : {genre_majoritaire_code}")

def main():
    st.set_page_config(page_title="Dashboard")

    st.markdown("<div style='text-align: center;'><h1>👋 Bienvenue sur l'outil de scoring crédit de Prêt à dépenser</h1></div>", unsafe_allow_html=True)

    # Charger les données du DataFrame df_final_test
    df_final_test_limited = pd.read_csv('df_final_test_limited.csv')

    # Remplacer les valeurs NaN et infinies par une valeur valide (par exemple, 0)
    df_final_test_limited = df_final_test_limited.replace([np.inf, -np.inf, np.nan], 0)

    # Charger votre modèle XGBoost entraîné
    model = xgb.XGBClassifier()
    model.load_model('mon_modele.json')

    # Permettre à l'utilisateur de saisir l'ID du client
    id_client = st.number_input("Entrez l'ID du client", min_value=0, step=1)

    # Afficher les informations du client et récupérer les caractéristiques sélectionnées
    caracteristiques_selectionnees = afficher_infos_client(id_client, df_final_test_limited)

    if st.button('Prédire'):
        if id_client < 0 or id_client >= len(df_final_test_limited):
            st.error("ID de client invalide.")
        else:
            # Sélectionner l'instance de test correspondant à l'ID du client saisi
            instance_data = df_final_test_limited.iloc[id_client].values.tolist()
            
            # Convertir les valeurs booléennes en entiers
            instance_data = [int(x) if isinstance(x, bool) else x for x in instance_data]

            reponse = send_request({'instance_data': instance_data})

            if reponse is None:
                st.warning("Aucune prédiction disponible.")
            else:
                prediction = reponse['prediction']
                probabilite_prediction = reponse['prediction_proba']
                score = probabilite_prediction[1] if prediction == 1 else probabilite_prediction[0]
                afficher_score(score, probabilite_prediction)
                afficher_caracteristiques_importantes(instance_data, df_final_test_limited, model)

    groupe_similaire = st.checkbox("Comparer à un groupe de clients similaires")
    if st.button("Comparer les informations"):
        comparer_infos(id_client, df_final_test_limited, groupe_similaire, caracteristiques_selectionnees)

if __name__ == '__main__':
    main()