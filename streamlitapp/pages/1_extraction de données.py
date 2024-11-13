# data_extraction_page.py

import streamlit as st
import requests

st.title("Extraction de données")

# Choix de la méthode d'extraction
extraction_method = st.selectbox("Choisissez la méthode d'extraction de données", ["Depuis un fichier", "Depuis S3", "Depuis PostgreSQL"])

# Répertoire de sauvegarde des fichiers extraits
save_directory = "data/uploaded_files"

if extraction_method == "Depuis un fichier":
    # Extraction depuis un fichier local
    st.subheader("Extraction depuis un fichier local")
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV ou Excel", type=["csv", "xlsx"])

    if st.button("Extraire depuis le fichier") and uploaded_file:
        # Enregistrer temporairement le fichier
        with open("temp_file_extract.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Appeler l'API pour extraire les données du fichier
        with open("temp_file_extract.csv", "rb") as f:
            response = requests.post(
                "http://localhost:8000/extract_file_data",
                files={"file": f}
            )

        if response.status_code == 200:
            st.success(f"Fichier extrait avec succès ! Sauvegardé à : {response.json()['file_path']}")
        else:
            st.error(f"Erreur lors de l'extraction depuis le fichier : {response.json()['detail']}")

elif extraction_method == "Depuis S3":
    # Extraction depuis un bucket S3
    st.subheader("Extraction depuis S3")
    bucket_name = st.text_input("Nom du bucket S3")
    file_name = st.text_input("Nom du fichier dans le bucket S3")

    if st.button("Extraire depuis S3") and bucket_name and file_name:
        response = requests.post(
            "http://localhost:8000/extract_s3_data",
            data={"bucket_name": bucket_name, "file_name": file_name}
        )

        if response.status_code == 200:
            st.success(f"Fichier extrait depuis S3 et sauvegardé à : {response.json()['file_path']}")
        else:
            st.error(f"Erreur lors de l'extraction depuis S3 : {response.json()['detail']}")

elif extraction_method == "Depuis PostgreSQL":
    # Extraction depuis une base de données PostgreSQL
    st.subheader("Extraction depuis PostgreSQL")
    query = st.text_area("Requête SQL pour extraire les données")

    if st.button("Extraire depuis PostgreSQL") and query:
        response = requests.post(
            "http://localhost:8000/extract_db_data",
            json={"query": query}
        )

        if response.status_code == 200:
            st.success(f"Données extraites depuis la base de données et sauvegardées à : {response.json()['file_path']}")
        else:
            st.error(f"Erreur lors de l'extraction depuis PostgreSQL : {response.json()['detail']}")
