import tempfile
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TRAINING_API_URL = "http://localhost:8000/train_xgboost_model"
METRICS_API_URL = "http://localhost:8000/metrics"

separators = r'[;,|]\s*'


# Fonction pour lire le fichier
def read_user_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, sep=separators)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Le fichier doit être au format .csv ou .xlsx")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


def get_most_correlated_feature(df: pd.DataFrame, target_column: str):
    correlations = df.corr()[target_column].drop(target_column)  # Supprimer la cible de la série de corrélation
    most_correlated = correlations.abs().idxmax()  # Trouver la colonne la plus corrélée
    return most_correlated, correlations[most_correlated]  # Retourner la colonne et la valeur de corrélation


# Fonction pour afficher le DataFrame sans Arrow via un CSV temporaire
def display_csv_as_dataframe(df: pd.DataFrame):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_df = pd.read_csv(tmp.name)  # Recharger le DataFrame depuis le CSV temporaire
    st.dataframe(tmp_df)  # Afficher sans Arrow


# Fonction regroupant toutes les visualisations et conversions pour Streamlit
def display_all_eda(df: pd.DataFrame) -> None:
    display_shape(df)
    display_random_rows(df)
    display_dtypes_and_na(df)
    barplot_missing_values(df)
    heatmap_missing_values(df)
    emptiest_rows(df)
    display_df_describe(df)
    correlation_heatmap(df)
    check_if_correlations(df)


def send_training_request(uploaded_file, target_column):
    # Vérifier que le fichier a des données
    if uploaded_file.size == 0:
        st.error("Le fichier téléversé est vide.")
        return

    # Réinitialiser le pointeur au début du fichier
    uploaded_file.seek(0)

    # Inclure le contenu du fichier dans `files`
    files = {"file": (uploaded_file.name, uploaded_file.read(), "multipart/form-data")}
    data = {"target_column": target_column}  # Passer la cible détectée à l'API

    try:
        response = requests.post(TRAINING_API_URL, files=files, data=data, timeout=300)
        if response.status_code == 200:
            st.success("Entraînement lancé avec succès !")
        else:
            try:
                error_detail = response.json().get("detail")
            except ValueError:
                error_detail = response.text
            st.error(f"Erreur : {response.status_code} - {error_detail}")
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion : {e}")


def display_shape(df: pd.DataFrame) -> None:
    st.subheader("Shape of the Dataset")
    st.write("There are", df.shape[0], "rows and", df.shape[1], "columns in the dataset.")


def display_random_rows(df: pd.DataFrame) -> None:
    st.subheader("Sample of 5 Random Rows from the Dataset")
    display_csv_as_dataframe(df.sample(5))


def display_dtypes_and_na(df: pd.DataFrame) -> None:
    st.subheader("Data Types and Missing Values by Column")
    column_names = list(df.columns)
    data_types = list(df.dtypes)
    missing_values = list(df.isnull().sum() / df.shape[0] * 100)
    df_df = pd.DataFrame({"Column Name": column_names, "Data Type": data_types, "Missing Values (%)": missing_values})
    display_csv_as_dataframe(df_df)


def heatmap_missing_values(df: pd.DataFrame) -> None:
    st.subheader("Heatmap of Missing Values by Column")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#f7e8a1")
    fig.set_facecolor("#eb804c")
    sns.heatmap(df.isnull(), cmap=["#f7e8a1", "#7a88cc"], cbar=False, ax=ax)
    plt.xticks(rotation=75)
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Features")
    ax.set_ylabel("Rows")
    st.pyplot(fig)


def barplot_missing_values(df: pd.DataFrame) -> None:
    st.subheader("Bar Plot of Missing Values by Column")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#f7e8a1")
    fig.set_facecolor("#eb804c")
    sns.barplot(x=df.columns, y=df.isnull().sum(), ax=ax)
    plt.xticks(rotation=75)
    plt.title("Missing Values by Column")
    plt.xlabel("Features")
    plt.ylabel("Sum of Missing Values")
    st.pyplot(fig)


def emptiest_rows(df: pd.DataFrame) -> None:
    st.subheader("Rows with the Most Missing Values")
    missing_values = df.isna().sum(axis=1)
    df_sorted = df.loc[missing_values.sort_values(ascending=False).index]
    df_top_5 = df_sorted.head(5)
    display_csv_as_dataframe(df_top_5)


def display_df_describe(df: pd.DataFrame) -> None:
    st.subheader("df.describe(); Summary Statistics of the Dataset")
    display_csv_as_dataframe(df.describe())


def correlation_heatmap(df):
    st.subheader("Correlation Heatmap")

    # Sélectionner uniquement les colonnes numériques
    num_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()

    # Vérifier qu'il y a au moins deux colonnes numériques pour calculer la corrélation
    if len(num_cols) < 1:
        st.warning("Pas assez de colonnes numériques pour afficher une carte de corrélation.")
        return

    # Calculer la matrice de corrélation
    corr = df[num_cols].corr()

    # Vérifier si la matrice de corrélation contient des valeurs
    if corr.empty:
        st.warning("La matrice de corrélation est vide. Impossible d'afficher la carte de corrélation.")
        return

    # Afficher la heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#f7e8a1")
    fig.set_facecolor("#eb804c")
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    plt.xticks(rotation=75)
    ax.set_title("Correlation Heatmap")
    ax.set_xlabel("Features")
    ax.set_ylabel("Rows")
    st.pyplot(fig)


def check_if_correlations(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
    if len(num_cols) < 2:
        st.warning("At least two numerical columns are required.")
        return
    corr = df[num_cols].corr()
    max_corr = corr.abs().unstack().sort_values(ascending=False).drop_duplicates()
    max_corr = max_corr[max_corr != 1]
    max_corr = max_corr[max_corr.abs() >= 0.7]
    if not max_corr.empty:
        col1, col2 = max_corr.index[0]
        prediction = f"I might be able to predict future {col1} using {col2}" if corr.loc[
                                                                                     col1, col2] > 0 else f"I might be able to predict future {col2} using {col1}"
        st.write(f"Highest correlation: {corr.loc[col1, col2]:.2f}")
        st.write(prediction)
        if len(max_corr) > 1:
            col3, col4 = max_corr.index[1]
            prediction = f"I might be able to predict future {col3} using {col4}" if corr.loc[
                                                                                         col3, col4] > 0 else f"I might be able to predict future {col4} using {col3}"
            st.write(f"Second highest correlation: {corr.loc[col3, col4]:.2f}")
            st.write(prediction)
    else:
        st.write("No correlations found.")
        mid_corr = corr.abs().unstack().sort_values(ascending=False).drop_duplicates()
        mid_corr = mid_corr[mid_corr != 1]
        mid_corr = mid_corr[(mid_corr.abs() >= 0.51) & (mid_corr.abs() < 0.7)]
        if not mid_corr.empty:
            for col5, col6 in mid_corr.index:
                st.write(f"Correlation between {col5} and {col6}: {corr.loc[col5, col6]:.2f}")
                st.write(
                    f"There is a correlation between {col5} and {col6}, but it's not strong enough to be sure if I can use one to predict the other.")
