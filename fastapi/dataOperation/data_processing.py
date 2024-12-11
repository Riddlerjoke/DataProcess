import logging
import shutil

import pandas as pd
from sqlalchemy import create_engine
from typing import List, Optional
from fastapi import UploadFile, HTTPException
import os


def preprocess_data(file_path: str, target_column: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        df = df.loc[:, df.isnull().mean() < 0.2]  # Supprime les colonnes avec plus de 20% de valeurs manquantes
        df.fillna(df.mode().iloc[0], inplace=True)  # Remplir valeurs manquantes
        df[target_column].fillna(df[target_column].mode()[0], inplace=True)  # Remplir NaN de la cible
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prétraitement des données : {e}")


def process_files(files: List[UploadFile], session_dir: str, merge_key: Optional[str] = None):
    dataframes = []
    for file in files:
        file_path = os.path.join(session_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        df = pd.read_csv(file_path) if file.filename.endswith('.csv') else pd.read_excel(file_path)
        dataframes.append(df)

    if merge_key:
        # Vérifier que la clé de fusion existe dans tous les fichiers
        for i, df in enumerate(dataframes):
            if merge_key not in df.columns:
                raise ValueError(f"La clé '{merge_key}' est absente dans le fichier {files[i].filename}.")

        # Fusionner les DataFrames
        merged_df = dataframes[0]
        for df in dataframes[1:]:
            merged_df = merged_df.merge(df, on=merge_key, how="outer")

        # Supprimer les doublons éventuels après fusion
        merged_df = merged_df.drop_duplicates()
    else:
        raise ValueError("Aucune clé de fusion spécifiée. Impossible de fusionner les fichiers.")

    # Sauvegarder le fichier fusionné
    merged_path = os.path.join(session_dir, "merged_data.csv")
    merged_df.to_csv(merged_path, index=False)
    logging.info(f"Données fusionnées sauvegardées à {merged_path}")
    return {"merged_file": merged_path}


def load_and_clean_file(file_path: str, column_mapping: Optional[dict] = None) -> pd.DataFrame:
    """
    Charge un fichier CSV ou Excel, nettoie les colonnes et les mappe aux noms attendus.
    """
    try:
        # Charger le fichier
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, sep=',', quotechar='"', engine='python')
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Format de fichier non pris en charge.")

        # Nettoyage des noms de colonnes
        if column_mapping:
            df.rename(columns=column_mapping, inplace=True)

        # Convertir les colonnes catégoriques
        categorical_columns = df.select_dtypes(include=["object"]).columns
        for col in categorical_columns:
            df[col] = df[col].astype("category")

        logging.info(f"Fichier chargé et nettoyé avec succès. Aperçu des données :\n{df.head()}")
        return df

    except Exception as e:
        logging.error(f"Erreur pendant le chargement et le nettoyage du fichier : {e}")
        raise ValueError(f"Erreur pendant le chargement et le nettoyage du fichier : {e}")


def preprocess_files(file_paths: List[str], merge_key: Optional[str] = None) -> pd.DataFrame:
    """Prépare les fichiers en les chargeant, fusionnant et nettoyant."""
    dataframes = [load_and_clean_file(file) for file in file_paths]
    if merge_key:
        merged_df = dataframes[0]
        for df in dataframes[1:]:
            if merge_key not in df.columns or merge_key not in merged_df.columns:
                raise HTTPException(status_code=400, detail=f"Clé de fusion '{merge_key}' manquante.")
            merged_df = merged_df.merge(df, on=merge_key, how="outer")
    else:
        common_columns = set(dataframes[0].columns).intersection(*(df.columns for df in dataframes[1:]))
        if not common_columns:
            raise HTTPException(status_code=400, detail="Aucune colonne commune pour effectuer la fusion.")
        merged_df = dataframes[0]
        for df in dataframes[1:]:
            merged_df = merged_df.merge(df, on=list(common_columns), how="outer")

    merged_df.dropna(inplace=True)
    for col in merged_df.select_dtypes(include=["datetime"]).columns:
        merged_df[col] = pd.to_datetime(merged_df[col], errors="coerce")
        merged_df.dropna(subset=[col], inplace=True)

    return merged_df


def aggregate_data(df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    """Agrège les données en fonction d'une clé de groupe."""
    if group_key in df.columns:
        return df.groupby(group_key).sum(numeric_only=True)
    else:
        return df.select_dtypes(include=['number']).sum().to_frame().T


def save_data(df: pd.DataFrame, directory: str, filename: str) -> str:
    """Sauvegarde les données et retourne le chemin du fichier sauvegardé."""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    df.to_csv(file_path, index=False)
    return file_path


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les noms des colonnes pour qu'ils correspondent aux colonnes attendues.
    """
    column_mapping = {
        'gender': 'Gender',
        'age': 'Age',
        'occupation': 'Occupation',
        'sleepduration': 'Sleep Duration',
        'qualityofsleep': 'Quality of Sleep',
        'physicalactivitylevel': 'Physical Activity Level',
        'stresslevel': 'Stress Level',
        'bmicategory': 'BMI Category',
        'bloodpressure': 'Blood Pressure',
        'heartrate': 'Heart Rate',
        'dailysteps': 'Daily Steps',
        'sleepdisorder': 'Sleep Disorder',
    }

    # Convertir les colonnes en minuscules et les renommer
    data.columns = data.columns.str.lower().str.replace(' ', '').str.replace('_', '').str.replace('-', '')
    data = data.rename(columns=column_mapping)

    return data


def preprocess_for_prediction(file_path: str, required_columns: Optional[list] = None) -> pd.DataFrame:
    """
    Charge et prépare un fichier pour la prédiction.
    """
    try:
        # Charger et nettoyer le fichier
        df = load_and_clean_file(file_path)

        # Normaliser les noms de colonnes
        df = normalize_columns(df)
        logging.info(f"Colonnes normalisées : {df.columns.tolist()}")

        # Vérifier la validité des colonnes
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Colonnes manquantes : {missing_columns}")

            expected_columns = [
                'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Physical Activity Level',
                'Stress Level', 'BMI Category', 'Blood Pressure', 'Heart Rate', 'Daily Steps', 'Sleep Disorder'
            ]

            # Vérifier les colonnes manquantes
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Colonnes manquantes : {missing_columns}")

            # Réordonner les colonnes
            df = df[expected_columns]

        logging.info("Prétraitement des données pour la prédiction terminé.")
        return df
    except Exception as e:
        logging.error(f"Erreur pendant le prétraitement pour la prédiction : {e}")
        raise ValueError(f"Erreur pendant le prétraitement pour la prédiction : {e}")
