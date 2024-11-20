import streamlit as st
import requests


st.title("Lien vers la Dernière Exécution de Modèle sur MLflow")

# Appel API pour récupérer le lien de la dernière exécution MLflow
try:
    response = requests.get("http://localhost:8000/last_mlflow_run_link")

    if response.status_code == 200:
        data = response.json()
        mlflow_run_link = data.get("mlflow_run_link", None)

        if mlflow_run_link:
            st.success("Lien vers la dernière exécution MLflow trouvé !")
            st.write("Cliquez sur le lien ci-dessous pour voir les détails de la dernière exécution :")
            st.markdown(f"[Voir la dernière exécution dans MLflow]({mlflow_run_link})", unsafe_allow_html=True)
        else:
            st.warning("Aucune exécution trouvée pour l'expérience.")
    else:
        st.error(f"Erreur lors de la récupération du lien MLflow : {response.json().get('detail', 'Erreur inconnue')}")
except Exception as e:
    st.error(f"Erreur de connexion à l'API : {e}")

st.title("Informations sur le Dernier Modèle Entraîné")

# Appeler l'API pour obtenir les informations du modèle
response = requests.get("http://localhost:8000/last_model_info")

if response.status_code == 200:
    model_info = response.json()
    st.subheader("Détails du Dernier Modèle")
    st.write(f"**Run ID** : {model_info['run_id']}")
    st.write(f"**Experiment ID** : {model_info['experiment_id']}")
    st.write(f"**Start Time** : {model_info['start_time']}")
    st.write(f"**Artifact URI** : {model_info['artifact_uri']}")

    st.subheader("Métriques")
    metrics_descriptions = {
        "accuracy": "Précision - proportion de prédictions correctes parmi toutes les prédictions.",
        "precision": "Précision - proportion de prédictions positives correctes par rapport au total des prédictions positives.",
        "recall": "Rappel - proportion de vrais positifs correctement identifiés parmi tous les vrais positifs.",
        "f1_score": "F1 Score - moyenne harmonique de la précision et du rappel, mesurant l'équilibre entre les deux.",
        "auc": "AUC (Area Under Curve) - mesure de la capacité du modèle à séparer les classes, basée sur la courbe ROC."
    }

    for metric, value in model_info["metrics"].items():
        description = metrics_descriptions.get(metric, "Aucune description disponible.")
        st.write(f"**{metric}** : {value} - {description}")

    st.subheader("Paramètres")
    for param, value in model_info["params"].items():
        st.write(f"**{param}** : {value}")

    # Lien vers l'exécution dans MLflow
    st.markdown(f"[Voir l'exécution dans MLflow]({model_info['mlflow_run_link']})", unsafe_allow_html=True)

else:
    st.error("Erreur pendant la récupération des informations du modèle.")
