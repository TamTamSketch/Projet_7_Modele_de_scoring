import numpy as np
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import shap
import xgboost as xgb
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.model_selection import cross_val_score

# Fonction pour convertir les types de données en types sérialisables
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

# Fonction pour envoyer une requête à l'API de prédiction
def send_request(features):
    url = 'http://13.60.25.39:5000/predict'
    instance_data = [convert_to_serializable(x) for x in features['instance_data']]
    data = {'data': [instance_data]}
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la requête : {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Contenu de la réponse : {e.response.text}")
        return None

# Fonction pour afficher le score du client
def afficher_score(score, probabilite_prediction):
    st.subheader("Score du client")
    
    score = float(score[0] if isinstance(score, (list, np.ndarray)) else score)

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
                {'range': [0.5, 1], 'color': "#FFA500"}
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
    fig.add_trace(go.Scatter(x=[score*100, score*100], y=[0, 1], mode='lines', line=dict(color='red', width=3, dash='dash'), name='Score du client'), row=1, col=2)

    fig.update_layout(height=400, width=800, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(title_text="Score", range=[0, 100], row=1, col=2)
    fig.update_yaxes(title_text="Niveau de risque", range=[0, 1], row=1, col=2)

    st.plotly_chart(fig)

    proba = float(probabilite_prediction[0] if isinstance(probabilite_prediction, (list, np.ndarray)) else probabilite_prediction)
    interpretation = f"Le client {'n''est pas' if score < 0.5 else 'est'} à risque pour emprunter. (Probabilité : {proba:.4f})"
    couleur = "lightblue" if score < 0.5 else "#FFA500"
    
    st.markdown(f'<p style="background-color:{couleur};padding:10px;border-radius:5px;color:black;">{interpretation}</p>', unsafe_allow_html=True)

# Fonction pour afficher les informations du client
def afficher_infos_client(id_client, df):
    client = df.iloc[id_client]
    st.subheader("Informations descriptives")

    colonnes = df.columns.tolist()
    colonnes_selectionnees = st.multiselect("Sélectionnez les informations à afficher", colonnes)

    if colonnes_selectionnees:
        st.write(client[colonnes_selectionnees])
    else:
        st.write("Aucune information sélectionnée.")

    return colonnes_selectionnees

# Fonction pour afficher les caractéristiques importantes
def afficher_caracteristiques_importantes(instance_data, df, model):
    st.subheader("Caractéristiques les plus impactantes")

    explainer = shap.TreeExplainer(model)
    instance_2d = np.array(instance_data).reshape(1, -1)
    shap_values_instance = explainer.shap_values(instance_2d)
    feature_names = df.columns.tolist()

    if isinstance(shap_values_instance, list):
        shap_values_instance = shap_values_instance[1]
    elif isinstance(shap_values_instance, np.ndarray) and shap_values_instance.ndim > 2:
        shap_values_instance = shap_values_instance[0]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[-1]

    plt.figure(figsize=(20, 3))
    shap.force_plot(expected_value, shap_values_instance, instance_2d, feature_names=feature_names, matplotlib=True, show=False)
    
    st.pyplot(plt.gcf())
    plt.clf()

    shap_values_sorted = sorted(zip(feature_names, shap_values_instance[0]), key=lambda x: abs(x[1]), reverse=True)
    st.write("Les 10 caractéristiques les plus impactantes pour l'instance :")
    for feature, shap_value in shap_values_sorted[:10]:
        st.write(f"{feature}: {shap_value:.3f}")

# Fonction pour comparer les informations du client
def comparer_infos(id_client, df, groupe_similaire, caracteristiques_selectionnees):
    client = df.iloc[id_client]
    if groupe_similaire:
        groupe = df[(df['EXT_SOURCE_1'] == client['EXT_SOURCE_1']) &
                    (df['EXT_SOURCE_2'] == client['EXT_SOURCE_2']) &
                    (df['EXT_SOURCE_3'] == client['EXT_SOURCE_3']) &
                    (df['CODE_GENDER'] == client['CODE_GENDER'])]
    else:
        groupe = df

    fig = go.Figure()

    for caracteristique in caracteristiques_selectionnees:
        fig.add_trace(go.Bar(x=['Client'], y=[client[caracteristique]], name=caracteristique))
        fig.add_trace(go.Bar(x=['Groupe'], y=[groupe[caracteristique].mean()], name=f"{caracteristique} Groupe"))

    fig.update_layout(barmode='group')
    st.plotly_chart(fig)

    code_gender_counts = groupe['CODE_GENDER'].value_counts()
    genre_majoritaire_code = code_gender_counts.idxmax()
    st.write(f"Le genre majoritaire pour le groupe est : {genre_majoritaire_code}")

# Fonction de coût métier
def cout_metier(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp + 5 * fn

# Créer un scorer personnalisé
cout_metier_scorer = make_scorer(cout_metier, greater_is_better=False)

# Fonction pour évaluer le modèle
def evaluer_modele(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring=cout_metier_scorer)
    return -np.mean(scores)  # On retourne la moyenne négative car le scorer est "greater_is_better=False"

# Fonction principale
def main():
    st.set_page_config(page_title="Dashboard de Scoring Crédit")

    st.markdown("<div style='text-align: center;'><h1>👋 Bienvenue sur l'outil de scoring crédit de Prêt à dépenser</h1></div>", unsafe_allow_html=True)

    # Chargement des deux DataFrames
    df_final_test_limited = pd.read_csv('df_final_test_limited.csv')
    df_final_test_limited = df_final_test_limited.replace([np.inf, -np.inf, np.nan], 0)

    df_final_train = pd.read_csv('df_final_train.csv')
    df_final_train = df_final_train.replace([np.inf, -np.inf, np.nan], 0)

    model = xgb.XGBClassifier()
    model.load_model('mon_modele.json')

    id_client = st.number_input("Entrez l'ID du client", min_value=0, max_value=len(df_final_test_limited)-1, step=1)

    caracteristiques_selectionnees = afficher_infos_client(id_client, df_final_test_limited)

    if st.button('Prédire'):
        instance_data = df_final_test_limited.iloc[id_client].values.tolist()
        instance_data = [int(x) if isinstance(x, bool) else x for x in instance_data]

        reponse = send_request({'instance_data': instance_data})

        if reponse is None:
            st.warning("Aucune prédiction disponible.")
        else:
            prediction = reponse['prediction']
            probabilite_prediction = reponse['prediction_proba']
            
            score = probabilite_prediction[1] if isinstance(probabilite_prediction, (list, np.ndarray)) and len(probabilite_prediction) > 1 else probabilite_prediction
            
            afficher_score(score, probabilite_prediction)
            afficher_caracteristiques_importantes(instance_data, df_final_test_limited, model)

    groupe_similaire = st.checkbox("Comparer à un groupe de clients similaires")
    if st.button("Comparer les informations"):
        comparer_infos(id_client, df_final_test_limited, groupe_similaire, caracteristiques_selectionnees)

    # Partie pour l'évaluation du modèle
    st.sidebar.header("Fiabilité du modèle")
    if st.sidebar.button('Vérifier la fiabilité du modèle'):
        X = df_final_train.drop('TARGET', axis=1)
        y = df_final_train['TARGET']
        
        cout_moyen = evaluer_modele(model, X, y)
        
        st.sidebar.subheader("Fiabilité du modèle")
        
        # Définissez des seuils pour l'interprétation du coût
        seuil_excellent = 5000
        seuil_bon = 10000
        seuil_moyen = 15000

        if cout_moyen <= seuil_excellent:
            niveau_confiance = "Très élevé"
            couleur = "green"
            conseil = "Vous pouvez avoir une grande confiance dans les prédictions du modèle."
        elif cout_moyen <= seuil_bon:
            niveau_confiance = "Élevé"
            couleur = "lightgreen"
            conseil = "Le modèle est fiable, mais vérifiez les cas limites."
        elif cout_moyen <= seuil_moyen:
            niveau_confiance = "Moyen"
            couleur = "orange"
            conseil = "Utilisez le modèle comme guide, mais appuyez-vous davantage sur votre expertise."
        else:
            niveau_confiance = "Faible"
            couleur = "red"
            conseil = "Soyez très prudent avec les prédictions du modèle. Votre jugement est crucial."

        st.sidebar.markdown(f"<h3 style='color:{couleur}'>Niveau de confiance : {niveau_confiance}</h3>", unsafe_allow_html=True)
        
        st.sidebar.write(f"Conseil pour le conseiller : {conseil}")
        
        st.sidebar.write("Comment utiliser cette information :")
        st.sidebar.write("1. Regardez d'abord le score de risque du client.")
        st.sidebar.write("2. Comparez ce score au niveau de confiance du modèle ci-dessus.")
        st.sidebar.write("3. Si le niveau de confiance est élevé, vous pouvez vous fier davantage au score.")
        st.sidebar.write("4. Si le niveau de confiance est faible, soyez plus prudent et appuyez-vous davantage sur votre analyse personnelle.")
        st.sidebar.write("5. Dans tous les cas, utilisez votre jugement professionnel et les informations complémentaires du dossier.")

if __name__ == '__main__':
    main()