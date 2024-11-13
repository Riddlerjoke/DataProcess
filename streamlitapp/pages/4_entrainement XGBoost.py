# 4_entrainement XGBoost.py

import streamlit as st
import requests

st.title("Entraînement du modèle XGBoost")

uploaded_file = st.file_uploader("Télécharger un fichier CSV pour l'entraînement", type="csv")
target_column = st.text_input("Nom de la colonne cible")

if st.button("Lancer l'entraînement"):
    if uploaded_file and target_column:
        # Enregistrer temporairement le fichier
        with open("temp_train_file.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Appeler l'API pour entraîner le modèle
        with open("temp_train_file.csv", "rb") as f:
            response = requests.post(
                "http://localhost:8000/train_xgboost_model",
                files={"file": f},
                data={"target_column": target_column},
            )

        if response.status_code == 200:
            st.success("Entraînement terminé avec succès!")
        else:
            st.error(f"Erreur pendant l'entraînement: {response.json()['detail']}")
    else:
        st.warning("Veuillez télécharger un fichier et spécifier la colonne cible.")
