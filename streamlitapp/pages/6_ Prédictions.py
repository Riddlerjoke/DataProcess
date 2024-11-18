import streamlit as st
import requests
import pandas as pd

st.title("Prédictions avec un modèle ML pré-entraîné")

# Entrées utilisateur
bucket_name = st.text_input("Nom du bucket S3", "mlflow")
model_path = st.text_input("Chemin du modèle dans S3", "path/to/model.joblib")
uploaded_file = st.file_uploader("Téléchargez un fichier pour la prédiction", type=["csv", "xlsx"])

# Fonction pour interpréter les résultats
def interpret_predictions(predictions):
    """
    Interprète les résultats des prédictions pour un affichage convivial.
    """
    interpretation_mapping = {
        0: "Bonne santé du sommeil",
        1: "Apnée du sommeil détectée",
        2: "Insomnie détectée",
        3: "Syndrome des jambes sans repos détecté",
    }
    return [interpretation_mapping.get(pred, "Inconnu") for pred in predictions]

# Lancer la prédiction
if st.button("Lancer la prédiction"):
    if bucket_name and model_path and uploaded_file:
        # Sauvegarde temporaire du fichier localement
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Appel de l'API pour effectuer la prédiction
        with open(uploaded_file.name, "rb") as f:
            response = requests.post(
                "http://localhost:8000/predict",
                data={"bucket_name": bucket_name, "model_path": model_path},
                files={"file": f},
            )

        # Vérifiez la réponse
        if response.status_code == 200:
            predictions = response.json()["predictions"]
            st.success("Prédictions effectuées avec succès !")

            # Interprétation des résultats
            st.subheader("Résultats des prédictions :")
            interpreted_results = interpret_predictions(predictions)
            results_df = pd.DataFrame({
                "Index": range(1, len(predictions) + 1),
                "Classe Prévue": predictions,
                "Interprétation": interpreted_results,
            })
            st.write(results_df)

            # Ajouter une visualisation graphique
            st.subheader("Visualisation des résultats :")
            st.bar_chart(results_df["Classe Prévue"].value_counts())
        else:
            st.error(f"Erreur pendant la prédiction : {response.json().get('detail', 'Erreur inconnue')}")
    else:
        st.warning("Veuillez remplir tous les champs et télécharger un fichier.")
