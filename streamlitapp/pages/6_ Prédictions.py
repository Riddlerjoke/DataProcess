import requests
import streamlit as st

st.title("Prédictions avec un modèle ML pré-entraîné")

# Entrées utilisateur
bucket_name = st.text_input("Nom du bucket S3", "mlflow")
model_path = st.text_input("Chemin du modèle dans S3", "path/to/model.joblib")
uploaded_file = st.file_uploader("Téléchargez un fichier pour la prédiction", type=["csv", "xlsx"])

if st.button("Lancer la prédiction"):
    if bucket_name and model_path and uploaded_file:
        try:
            # Sauvegarder temporairement le fichier localement
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Préparer les données pour l'appel API
            with open(uploaded_file.name, "rb") as f:
                response = requests.post(
                    "http://localhost:8000/predict",
                    data={"bucket_name": bucket_name, "model_path": model_path},
                    files={"file": f},
                )

            # Vérifiez la réponse
            if response.status_code == 200:
                predictions = response.json()["predictions"]
                st.success("Prédictions terminées avec succès!")
                st.write(predictions)
            else:
                st.error(f"Erreur pendant la prédiction : {response.json()['detail']}")
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de connexion à l'API : {e}")
        except Exception as e:
            st.error(f"Erreur inattendue : {e}")
    else:
        st.warning("Veuillez remplir tous les champs et télécharger un fichier.")
