import os
import pandas as pd
from joblib import load
import boto3
from fastapi import HTTPException
import logging
from botocore.exceptions import NoCredentialsError, PartialCredentialsError


# Configuration de MinIO
s3_client = boto3.client(
    "s3",
    endpoint_url="http://minio:9000",
    aws_access_key_id=os.getenv('MINIO_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('MINIO_SECRET_ACCESS_KEY')
)


def load_model_from_s3(bucket_name: str, model_path: str):
    try:
        # Téléchargement du fichier depuis le bucket S3
        response = s3_client.get_object(Bucket=bucket_name, Key=model_path)
        temp_model_path = "temp_model.joblib"

        # Sauvegarde temporaire
        with open(temp_model_path, "wb") as f:
            f.write(response["Body"].read())

        # Vérifier l'existence du fichier
        if not os.path.exists(temp_model_path):
            raise FileNotFoundError(f"Le fichier {temp_model_path} n'a pas été correctement téléchargé.")

        # Charger le modèle
        model = load(temp_model_path)
        logging.info("Modèle chargé avec succès depuis S3.")
        return model
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Erreur d'accès à S3 : {e}")
        raise ValueError("Erreur d'accès à S3.")
    except FileNotFoundError as e:
        logging.error(f"Erreur de fichier : {e}")
        raise ValueError(f"Erreur de fichier : {e}")
    except Exception as e:
        logging.error(f"Erreur pendant le chargement du modèle : {e}")
        raise ValueError(f"Erreur pendant le chargement du modèle : {e}")

def preprocess_for_model(data: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    """
    Préparer les données pour qu'elles correspondent aux colonnes attendues par le modèle.
    """
    try:
        # Normaliser les noms des colonnes
        data.columns = (
            data.columns.str.strip()
                         .str.lower()
                         .str.replace(' ', '')
                         .str.replace('_', '')
        )

        # Vérifier les colonnes manquantes
        missing_columns = set(expected_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Colonnes manquantes dans les données : {missing_columns}")

        # Réordonner les colonnes pour qu'elles correspondent au modèle
        data = data[expected_columns]

        # Encoder les colonnes catégoriques si nécessaire
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].astype('category').cat.codes

        logging.info("Prétraitement des données pour le modèle terminé.")
        return data
    except Exception as e:
        logging.error(f"Erreur pendant le prétraitement des données : {e}")
        raise ValueError(f"Erreur pendant le prétraitement des données : {e}")


def predict(data: pd.DataFrame, model, expected_columns: list) -> pd.DataFrame:
    """
    Effectuer une prédiction avec le modèle chargé.
    """
    try:
        # Vérifier et réorganiser les colonnes pour correspondre au modèle
        data = data[expected_columns]
        predictions = model.predict(data)
        logging.info("Prédictions effectuées avec succès.")
        return predictions
    except Exception as e:
        logging.error(f"Erreur pendant la prédiction : {e}")
        raise ValueError(f"Erreur pendant la prédiction : {e}")
