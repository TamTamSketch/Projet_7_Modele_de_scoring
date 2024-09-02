import numpy as np
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import shap
import xgboost as xgb
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

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
    url = 'http://13.60.25.39:5000/predict'
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
    
    if isinstance(score, (list, np.ndarray)):
        score = score[0]
    score = float(score)

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'indicator'}, {'type': 'scatter'}]])

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Score de risque"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 0.5], 'color': "lightblue"},
                {'range': [0.5, 1], 'color': "#FFA500"}  # Orange instead of yellow
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.5
            }
        }
    ), row=1, col=1)

    x = list(range(100))
    y_low = [0.25 + 0.1 * (i % 2) for i in x]
    y_high = [0.75 + 0.1 * ((i + 1) % 2) for i in x]

    fig.add_trace(go.Scatter(x=x, y=y_low, mode='lines', line=dict(color='blue', width=2), name='Faible risque'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=y_high, mode='lines', line=dict(color='#FFA500', width=2), name='Risque élevé'), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[score*100, score*100],
        y=[0, 1],
        mode='lines',
        line=dict(color='red', width=3, dash='dash'),
        name='Score du client'
    ), row=1, col=2)

    fig.update_layout(
        height=400, 
        width=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(title_text="Score", range=[0, 100], row=1, col=2)
    fig.update_yaxes(title_text="Niveau de risque", range=[0, 1], row=1, col=2)

    st.plotly_chart(fig)

    if isinstance(probabilite_prediction, (float, int)):
        proba = probabilite_prediction
    elif isinstance(probabilite_prediction, (list, tuple, np.ndarray)) and len(probabilite_prediction) > 0:
        proba = probabilite_prediction[0]
    else:
        proba = score

    if score < 0.5:
        interpretation = f"Le client n'est pas à risque pour emprunter. (Probabilité : {proba:.4f})"
        couleur = "lightblue"
    else:
        interpretation = f"Le client est à risque pour emprunter. (Probabilité : {proba:.4f})"
        couleur = "#FFA500"  # Orange instead of yellow
    
    st.markdown(f'<p style="background-color:{couleur};padding:10px;border-radius:5px;color:black;">{interpretation}</p>', unsafe_allow_html=True)

def afficher_infos_client(id_client, df_final_test_limited):
    client = df_final_test_limited.iloc[id_client]
    st.subheader("Informations descriptives")

    colonnes = df_final_test_limited.columns.tolist()
    colonnes_selectionnees = st.multiselect("Sélectionnez les informations à afficher", colonnes)

    if colonnes_selectionnees:
        st.write(client[colonnes_selectionnees])
    else:
        st.write("Aucune information sélectionnée.")

    return colonnes_selectionnees

def afficher_caracteristiques_importantes(instance_data, df_final_test_limited, model):
    st.subheader("Caractéristiques les plus impactantes")

    explainer = shap.TreeExplainer(model)
    instance_2d = np.array(instance_data).reshape(1, -1)
    shap_values_instance = explainer.shap_values(instance_2d)
    feature_names = df_final_test_limited.columns.tolist()

    # S'assurer que shap_values_instance est 2D
    if isinstance(shap_values_instance, list) and len(shap_values_instance) == 2:
        shap_values_instance = shap_values_instance[1]
    elif isinstance(shap_values_instance, np.ndarray) and shap_values_instance.ndim > 2:
        shap_values_instance = shap_values_instance[0]

    # Gérer le cas où expected_value est un scalaire
    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    elif isinstance(expected_value, list):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    # Si c'est déjà un scalaire (comme numpy.float32), on le laisse tel quel

    # Créer le plot SHAP
    plt.figure(figsize=(20, 3))
    shap.force_plot(
        expected_value,
        shap_values_instance,
        instance_2d,
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    
    # Capturer le plot et l'afficher dans Streamlit
    fig = plt.gcf()
    st.pyplot(fig)
    plt.clf()  # Nettoyer la figure matplotlib

    # Afficher les 10 valeurs SHAP les plus impactantes
    shap_values_sorted = sorted(zip(feature_names, shap_values_instance[0]), key=lambda x: abs(x[1]), reverse=True)
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

    for caracteristique in caracteristiques_selectionnees:
        fig.add_trace(go.Bar(x=['Client'], y=[client[caracteristique]], name=caracteristique))
        fig.add_trace(go.Bar(x=['Groupe'], y=[groupe[caracteristique].mean()], name=f"{caracteristique} Groupe"))

    fig.update_layout(barmode='group')
    st.plotly_chart(fig)

    code_gender_counts = groupe['CODE_GENDER'].value_counts()
    genre_majoritaire_code = code_gender_counts.idxmax()
    st.write(f"Le genre majoritaire pour le groupe est : {genre_majoritaire_code}")

def main():
    st.set_page_config(page_title="Dashboard")

    st.markdown("<div style='text-align: center;'><h1>👋 Bienvenue sur l'outil de scoring crédit de Prêt à dépenser</h1></div>", unsafe_allow_html=True)

    df_final_test_limited = pd.read_csv('df_final_test_limited.csv')
    df_final_test_limited = df_final_test_limited.replace([np.inf, -np.inf, np.nan], 0)

    model = xgb.XGBClassifier()
    model.load_model('mon_modele.json')

    id_client = st.number_input("Entrez l'ID du client", min_value=0, step=1)

    caracteristiques_selectionnees = afficher_infos_client(id_client, df_final_test_limited)

    if st.button('Prédire'):
        if id_client < 0 or id_client >= len(df_final_test_limited):
            st.error("ID de client invalide.")
        else:
            instance_data = df_final_test_limited.iloc[id_client].values.tolist()
            instance_data = [int(x) if isinstance(x, bool) else x for x in instance_data]

            reponse = send_request({'instance_data': instance_data})

            if reponse is None:
                st.warning("Aucune prédiction disponible.")
            else:
                prediction = reponse['prediction']
                probabilite_prediction = reponse['prediction_proba']
                
                if isinstance(probabilite_prediction, (list, np.ndarray)) and len(probabilite_prediction) > 1:
                    score = probabilite_prediction[1]
                else:
                    score = probabilite_prediction[0] if isinstance(probabilite_prediction, (list, np.ndarray)) else probabilite_prediction
                
                score = float(score)
                afficher_score(score, probabilite_prediction)
                afficher_caracteristiques_importantes(instance_data, df_final_test_limited, model)

    groupe_similaire = st.checkbox("Comparer à un groupe de clients similaires")
    if st.button("Comparer les informations"):
        comparer_infos(id_client, df_final_test_limited, groupe_similaire, caracteristiques_selectionnees)

if __name__ == '__main__':
    main()