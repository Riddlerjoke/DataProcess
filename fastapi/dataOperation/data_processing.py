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

    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = merged_df.merge(df, on=merge_key, how="outer") if merge_key else merged_df

    merged_path = os.path.join(session_dir, "merged_data.csv")
    merged_df.to_csv(merged_path, index=False)
    logging.info(f"Données fusionnées sauvegardées à {merged_path}")
    return {"merged_file": merged_path}


def load_and_clean_file(file_path: str, clean_columns: bool = True) -> pd.DataFrame:
    """Charge un fichier CSV ou Excel et nettoie les colonnes."""
    try:
        if file_path.endswith('.csv'):
            # Charge le fichier CSV avec un traitement pour les guillemets doubles
            df = pd.read_csv(file_path, sep=',', quotechar='"', engine='python')
            # Remplace les guillemets dans les cellules
            for col in df.select_dtypes(include='object').columns:
                df[col] = df[col].str.replace('"', '', regex=False)

        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Format de fichier non pris en charge.")

        # Nettoyage des noms de colonnes si l'option est activée
        if clean_columns:
            df.columns = (df.columns
                          .str.replace(r'[-_]', ' ', regex=True)
                          .str.strip()
                          .str.title()
                          .str.replace(' ', ''))
        return df

    except Exception as e:
        logging.error(f"Erreur lors du chargement du fichier : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de chargement du fichier : {e}")


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
