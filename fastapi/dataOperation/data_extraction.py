import os
import pandas as pd
import boto3
from dotenv import load_dotenv
from sqlalchemy import create_engine
from fastapi import HTTPException, UploadFile

load_dotenv()

s3_client = boto3.client(
    "s3",
    endpoint_url="http://minio:9000",
    aws_access_key_id=os.getenv('MINIO_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('MINIO_SECRET_ACCESS_KEY')
)
engine = create_engine(
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_DB')}")


def extract_file_data(file: UploadFile, save_dir: str) -> str:
    file_path = os.path.join(save_dir, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    return file_path


def extract_s3_data(bucket_name: str, file_name: str) -> str:
    # Définir le répertoire de sauvegarde par défaut
    save_dir = "data/uploaded_files"
    try:
        # Vérifier et créer le répertoire de sauvegarde si nécessaire
        os.makedirs(save_dir, exist_ok=True)

        # Extraction du fichier depuis S3
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_name)
        df = pd.read_csv(obj['Body'])

        # Chemin complet pour enregistrer le fichier extrait
        file_path = os.path.join(save_dir, "s3_data.csv")
        df.to_csv(file_path, index=False)

        return file_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'extraction S3 : {e}")


def extract_db_data(query: str, save_dir: str, table_name: str) -> str:
    try:
        df = pd.read_sql(query, engine)
        file_path = os.path.join(save_dir, f"{table_name}.csv")
        df.to_csv(file_path, index=False)
        return file_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'extraction depuis la base de données : {e}")
