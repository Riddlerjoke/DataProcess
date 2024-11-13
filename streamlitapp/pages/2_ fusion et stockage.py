import streamlit as st
import requests

st.set_page_config(page_title="Traitement et Insertion de Données", layout="wide")
st.title("Application de Traitement et d'Insertion de Données")

# Section 1: Traitement de fichiers multiples
st.header("Traitement de fichiers multiples")

uploaded_files = st.file_uploader("Téléchargez plusieurs fichiers", type=["csv", "xlsx"], accept_multiple_files=True)
merge_key = st.text_input("Clé de fusion (facultatif)")

if st.button("Lancer le traitement"):
    if uploaded_files:
        file_paths = []
        for uploaded_file in uploaded_files:
            # Enregistrer temporairement chaque fichier
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(temp_path)

        # Préparer les fichiers pour l'API
        files = [("files", (file, open(file, "rb"), "multipart/form-data")) for file in file_paths]
        data = {"merge_key": merge_key} if merge_key else {}

        # Appeler l'API pour le traitement
        response = requests.post("http://localhost:8000/process_multiple_files", files=files, data=data)

        # Fermer les fichiers ouverts
        for _, file_obj in files:
            file_obj[1].close()

        if response.status_code == 200:
            st.success("Traitement terminé! Fichiers fusionnés et agrégés sauvegardés.")
            st.write(response.json())
        else:
            st.error(f"Erreur pendant le traitement : {response.json()['detail']}")
    else:
        st.warning("Veuillez télécharger des fichiers pour le traitement.")

# Section 2: Insertion de données dans PostgreSQL
st.header("Insertion de données dans PostgreSQL")

uploaded_file = st.file_uploader("Téléchargez un fichier CSV pour l'insertion", type="csv", key="upload_csv")
table_name = st.text_input("Nom de la table (laissez vide pour utiliser le nom du fichier)", key="table_name")

if st.button("Insérer les données"):
    if uploaded_file:
        # Enregistrer temporairement le fichier
        with open("temp_postgres_file.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Appeler l'API pour insérer les données dans PostgreSQL
        with open("temp_postgres_file.csv", "rb") as f:
            data = {"table_name": table_name} if table_name else {}
            response = requests.post(
                "http://localhost:8000/upload_to_postgres",
                files={"file": f},
                data=data
            )

        if response.status_code == 200:
            table_name_display = table_name or uploaded_file.name
            st.success(f"Données insérées avec succès dans la table '{table_name_display}'.")
            st.write(response.json())
        else:
            st.error(f"Erreur pendant l'insertion : {response.json()['detail']}")
    else:
        st.warning("Veuillez télécharger un fichier CSV pour l'insertion.")
