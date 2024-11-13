from sklearn.preprocessing import LabelEncoder
from data.fonctions import *  # Importer toutes les fonctions de 'fonctions.py'
import streamlit as st
import pandas as pd


# Fonction pour encoder les colonnes non numériques
def encode_non_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode non-numeric columns to ensure compatibility for correlation calculation."""
    df_encoded = df.copy()
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df[column].astype(str))
    return df_encoded


def get_best_target_column(df: pd.DataFrame):
    """Automatically identify the target column based on correlation analysis."""
    df_encoded = encode_non_numeric_columns(df)  # Encode non-numeric columns
    correlations = df_encoded.corr().abs()  # Calculate absolute correlations
    target_column = correlations.sum().idxmax()  # Select column with highest sum of correlations
    most_correlated, correlation_value = get_most_correlated_feature(df_encoded, target_column)
    return target_column, most_correlated, correlation_value


# Interface principale de l'application
st.title("CSV File Uploader")
uploaded_file = st.file_uploader("Choisissez un fichier CSV ou Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Lire et afficher le contenu brut pour vérifier qu'il n'est pas vide
    file_content = uploaded_file.read()
    st.write("Contenu brut du fichier :")
    st.text(file_content[:500])  # Afficher les premiers 500 caractères pour vérification
    uploaded_file.seek(0)  # Remettre le pointeur au début après chaque lecture

    # Charger et afficher le fichier en DataFrame
    try:
        df = read_user_file(uploaded_file)
        st.write("Aperçu du fichier téléversé :")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
        st.stop()

    # Exécuter l'analyse EDA
    display_all_eda(df)

    # Identifier automatiquement la colonne cible et la plus corrélée
    try:
        target_column, most_correlated, correlation_value = get_best_target_column(df)
        st.write(f"Colonne cible détectée automatiquement pour l'entraînement : {target_column}")
        st.write(
            f"Colonne ayant la plus forte corrélation avec {target_column} : {most_correlated} (corrélation = {correlation_value:.2f})")
    except Exception as e:
        st.error(f"Erreur pendant le calcul de la cible ou de la corrélation : {e}")
        st.stop()

    # Bouton pour lancer l'entraînement
    if st.button("Lancer l'entraînement"):
        send_training_request(uploaded_file, target_column)
