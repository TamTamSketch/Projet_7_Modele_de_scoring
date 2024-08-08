Ce fichier a pour objectif de servir de fichier introductif permettant à son lecteur de comprendre l'objectif du projet et le découpage des dossiers.

Objectif projet :

    Créer un outil de scoring crédit pour calculer la probabilité qu'un client rembourse son crédit, puis classifier la demande en crédit accordé ou refusé.


Découpage des dossiers :

    Un dossier de code réalisé sur un google colab pour :

                                                - l'analyse exploratoire,
                                                - la préparation des données et le feature engineering,
                                                - entraînement, optimisation et simplification modèle de machine learning,
                                                - interprétation du modèle (globale et locale)
                                                - le tableau HTML d'analyse de data drift réalisé à partir d'evidently
    

    Un dossier de code géré sur poetry via l'outil de versioning GitHub pour :

                                                - l'API FLASK : fichier api.py
                                                - le dashboard STREAMLIT : fichier app_dashboard.py
                                                - le modele de machine learning : fichier mon_modele.json
                                                - les packages utilisés : fichier requirements.txt généré à l'aide du pyproject.toml
    

    Une note méthodologique est annexée au projet global.

    Le déploiement de l'application dashboard et de l'API ont été réalisées sur AWS.

