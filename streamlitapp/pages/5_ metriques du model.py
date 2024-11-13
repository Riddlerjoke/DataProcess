import streamlit as st
import requests

st.title("Métriques d'Entraînement du Modèle")

# Appel API pour récupérer les métriques
try:
    response = requests.get("http://localhost:8000/metrics")
    if response.status_code == 200:
        metrics_data = response.json().get("metrics", {})

        if metrics_data:
            st.subheader("Dernières métriques enregistrées :")
            # Afficher les métriques dans une liste
            for metric_name, metric_value in metrics_data.items():
                st.write(f"**{metric_name}**: {metric_value}")
        else:
            st.warning("Aucune métrique trouvée pour le dernier entraînement.")
    else:
        st.error(f"Erreur lors de la récupération des métriques : {response.json().get('detail', 'Erreur inconnue')}")
except Exception as e:
    st.error(f"Erreur de connexion à l'API : {e}")

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
