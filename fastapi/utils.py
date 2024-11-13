import logging
import os
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional

import boto3
from fastapi import FastAPI, UploadFile, File, Form
from dotenv import load_dotenv
from sqlalchemy import create_engine

from dataOperation.data_processing import preprocess_data, process_files
from training.train_xgboost_mlflow import train_xgboost_model

app = FastAPI()
DATA_DIR = "data/datajson"
UPLOAD_DIRECTORY = "data/uploaded_files"
CLEAN_DIR = "data/dataclean"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

load_dotenv()

# MinIO Configuration
s3_client = boto3.client(
    "s3",
    endpoint_url="http://minio:9000",
    aws_access_key_id=os.environ['MINIO_ACCESS_KEY'],
    aws_secret_access_key=os.environ['MINIO_SECRET_ACCESS_KEY']
)

# PostgreSQL Configuration
DB_USER = os.environ['POSTGRES_USER']
DB_PASSWORD = os.environ['POSTGRES_PASSWORD']
DB_HOST = "postgres"
DB_PORT = "5432"
DB_NAME = os.environ['POSTGRES_DB']
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

logging.basicConfig(level=logging.INFO)


@app.post("/train_xgboost_model")
async def train_model(file: UploadFile = File(...), target_column: str = Form(...)):
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Charger et prétraiter les données
    df = preprocess_data(file_path, target_column)
    train_xgboost_model(df, target_column)
    os.remove(file_path)  # Nettoyer le fichier après l'entraînement
    return {"status": "success", "message": "Entraînement du modèle terminé."}


@app.get("/metrics")
def get_metrics() -> Dict[str, Any]:
    # Récupération et affichage des métriques d'un modèle de MLflow
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Your Experiment Name Here")
    if not experiment:
        return {"error": "Experiment not found"}

    latest_run = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
    return {"metrics": latest_run[0].data.metrics} if latest_run else {"error": "No runs found"}


@app.post("/process_multiple_files")
async def process_multiple_files(files: List[UploadFile] = File(...), merge_key: Optional[str] = None):
    session_dir = os.path.join(CLEAN_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)
    results = process_files(files, session_dir, merge_key)
    return results
