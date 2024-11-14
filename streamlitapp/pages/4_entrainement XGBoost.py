import streamlit as st
import requests

st.title("Entraînement du modèle XGBoost")

# Téléchargement du fichier de données
uploaded_file = st.file_uploader("Télécharger un fichier CSV pour l'entraînement", type="csv")

# Spécification de la colonne cible
target_column = st.text_input("Nom de la colonne cible")

# Spécification d'un nom pour l'expérimentation
experiment_name = st.text_input("Nom de l'expérimentation", value="Default Experiment")

# Option pour charger un modèle existant
model_path = st.text_input("Chemin vers un modèle existant (facultatif)")

if st.button("Lancer l'entraînement"):
    if uploaded_file and target_column:
        # Sauvegarde temporaire du fichier CSV
        with open("temp_train_file.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Appel de l'API pour entraîner ou charger le modèle
        with open("temp_train_file.csv", "rb") as f:
            response = requests.post(
                "http://localhost:8000/train_xgboost_model",
                files={"file": f},
                data={
                    "target_column": target_column,
                    "experiment_name": experiment_name,
                    "model_path": model_path,
                },
            )

        # Gestion de la réponse de l'API
        if response.status_code == 200:
            st.success("Entraînement terminé avec succès!")
            st.info("Modèle sauvegardé à l'adresse : saved_models/xgb_model.joblib")
        else:
            st.error(f"Erreur pendant l'entraînement : {response.json().get('detail', 'Erreur inconnue')}")
    else:
        st.warning("Veuillez télécharger un fichier et spécifier la colonne cible.")
