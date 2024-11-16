import os
import pandas as pd
from joblib import load
import boto3
from fastapi import HTTPException
import logging


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

        # Sauvegarder temporairement le fichier localement
        temp_model_path = "temp_model.joblib"
        with open(temp_model_path, "wb") as f:
            f.write(response["Body"].read())

        # Charger le modèle depuis le fichier local
        model = load(temp_model_path)

        # Supprimer le fichier temporaire après chargement
        os.remove(temp_model_path)

        return model
    except Exception as e:
        raise RuntimeError(f"Erreur pendant le chargement du modèle : {e}")


def predict(data: pd.DataFrame, model) -> pd.DataFrame:
    """
    Effectuer une prédiction avec le modèle chargé.
    """
    try:
        predictions = model.predict(data)
        logging.info("Prédictions effectuées avec succès.")
        return predictions
    except Exception as e:
        logging.error(f"Erreur pendant la prédiction : {e}")
        raise ValueError(f"Erreur pendant la prédiction : {e}")
