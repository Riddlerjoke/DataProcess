from data.fonctions import *  # Importer toutes les fonctions de 'fonctions.py'
import streamlit as st
import pandas as pd

# Titre de l'application
st.title("CSV File Uploader")
uploaded_file = st.file_uploader("Choisissez un fichier CSV ou Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Lire et afficher le contenu brut pour vérifier qu'il n'est pas vide
    file_content = uploaded_file.read()
    st.write("Contenu brut du fichier :")
    st.text(file_content[:500])  # Afficher les premiers 500 caractères pour vérification
    uploaded_file.seek(0)  # Remettre le pointeur au début

    # Charger et afficher le fichier en DataFrame
    try:
        df = read_user_file(uploaded_file)
        st.write("Aperçu du fichier téléversé :")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
        st.stop()  # Arrêter si le fichier ne peut pas être lu

    # Exécuter l'analyse EDA
    display_all_eda(df)

    # Bouton pour lancer l'entraînement
    if st.button("Lancer l'entraînement"):
        send_training_request(uploaded_file)