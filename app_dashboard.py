import numpy as np
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import xgboost as xgb
import shap

def send_request(features):
    url = 'http://localhost:5000/predict'
    data = {'data': [features['instance_data']]}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erreur : {response.status_code}")
        return None

def main():
    st.set_page_config(page_title="Dashboard")

    st.markdown("<div style='text-align: center;'><h1>👋 Bienvenue sur l'outil de scoring crédit de Prêt à dépenser</h1></div>", unsafe_allow_html=True)

    # Charger les données du DataFrame df_final_test (faute de vrai client)
    df_final_test = pd.read_csv('/home/tamara-daniel-tricot/Bureau/0_Projets_ParcoursDataScientist/P7/Projet_7/df_final_test.csv')

    # Remplacer les valeurs NaN et infinies par une valeur valide (par exemple, 0)
    df_final_test = df_final_test.replace([np.inf, -np.inf, np.nan], 0)

    # Convertir le DataFrame en un tableau NumPy en selectionnant la 4eme ligne(soit un client)
    instance_data = df_final_test.iloc[3].values.reshape(1, -1).tolist()[0]

    # Convertir les valeurs booléennes en entiers (0 ou 1)
    instance_data = [int(x) for x in instance_data]

    if st.button('Prédire'):
        response = send_request({'instance_data': instance_data})

        with st.expander("Score du client"):
            if response is None:
                st.warning("Aucune prédiction disponible.")
            else:
                prediction = response['prediction']
                prediction_proba = response['prediction_proba']
                if prediction == 1:
                    st.error(f'Le client est à risque pour emprunter. (Probabilité : {prediction_proba[1]:.2f})')
                else:
                    st.success(f'Le client n\'est pas à risque pour emprunter. (Probabilité : {prediction_proba[0]:.2f})')

        # Charger votre modèle XGBoost entraîné
        model = xgb.Booster(model_file='/home/tamara-daniel-tricot/Bureau/0_Projets_ParcoursDataScientist/P7/Projet_7/mon_modele.json')

        # Charger les données d'entraînement
        df_final_train = pd.read_csv('/home/tamara-daniel-tricot/Bureau/0_Projets_ParcoursDataScientist/P7/Projet_7/df_final_train.csv')

        # Séparation des features et de la cible
        X_train = df_final_train.drop('TARGET', axis=1)
        y_train = df_final_train['TARGET']

        # Calculer les valeurs SHAP pour l'ensemble du jeu de données
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        # Visualiser l'interprétation locale pour l'instance spécifique
        instance = X_train.values[3]  # Sélectionner la même instance que pour la prédiction
        instance_label = y_train.values[3]  # Récupérer la valeur cible pour cette instance

        # Convertir l'instance en une matrice 2D
        instance_2d = instance.reshape(1, -1)

        # Obtenir les valeurs SHAP pour cette instance
        shap_values_instance = explainer.shap_values(instance_2d)

        # Aplatir le tableau 2D shap_values_instance en un tableau 1D
        shap_values_instance_1d = shap_values_instance.flatten()

        # Créer un DataFrame avec les valeurs SHAP et les noms de features
        shap_df = pd.DataFrame({'feature': X_train.columns, 'shap_value': shap_values_instance_1d})

        # Sélectionner les 10 features les plus importantes (en valeur absolue)
        top_features = pd.concat([shap_df.nlargest(10, 'shap_value'), shap_df.nsmallest(10, 'shap_value')])

        # Créer la trace Plotly
        trace = go.Bar(x=top_features['feature'], y=top_features['shap_value'],
                       marker=dict(color=top_features['shap_value'].apply(lambda x: 'red' if x > 0 else 'green')))

        # Configurer la mise en page
        if instance_label == 0:
            prediction_label = "emprunt non risqué"
        else:
            prediction_label = "emprunt à risque"

        layout = go.Layout(title=f"Contributions des Caractéristiques du Client (Prédiction: {prediction_label})",
                           xaxis=dict(title='Caractéristiques du Client'),
                           yaxis=dict(title='Valeur SHAP'),
                           legend=dict(x=0.8, y=1, traceorder='normal', font=dict(size=12),
                                        bgcolor='rgba(0,0,0,0)',
                                        bordercolor='Black',
                                        borderwidth=2),
                           annotations=[],
                           updatemenus=[
                               dict(
                                   type="buttons",
                                   buttons=[
                                       dict(label="Afficher la légende",
                                            method="update",
                                            args=[{"visible": [True, True, True]},
                                                  {"annotations": [dict(x=0.8, y=1.1, text="<b>Légende :</b>", showarrow=False, font=dict(size=12)),
                                                                  dict(x=0.8, y=1.05, text="<span style='color:green'>Vert : Contribution négative (emprunt non risqué)</span>", showarrow=False, font=dict(size=12)),
                                                                  dict(x=0.8, y=1, text="<span style='color:red'>Rouge : Contribution positive (emprunt à risque)</span>", showarrow=False, font=dict(size=12))]}
                                           ]),
                                       dict(label="Masquer la légende",
                                            method="update",
                                            args=[{"visible": [True, False, False]},
                                                  {"annotations": []}
                                           ])
                                   ]
                               )
                           ])

        # Créer la figure Plotly
        fig = go.Figure(data=[trace], layout=layout)

        # Afficher la figure
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()