import logging
import os
import shutil
from datetime import datetime

import mlflow
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import List, Optional

from pydantic import BaseModel
from sqlalchemy import create_engine

from dataOperation.data_processing import preprocess_files, aggregate_data, save_data
from dataOperation.data_extraction import extract_file_data, extract_s3_data, extract_db_data
from training.train_xgboost_mlflow import train_xgboost_model
from postgres.ajoutpostgre import add_to_postgres, clean_table_name

app = FastAPI()

# Configuration des répertoires
DATA_DIR = "data/uploaded_files"
CLEAN_DIR = "data/cleaned_files"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)

engine = create_engine(
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_DB')}")


class DBExtractionRequest(BaseModel):
    query: str
    table_name: Optional[str] = "extracted_data"


################ l'entraînement XGBoost ################
@app.post("/train_xgboost_model")
async def train_model(file: UploadFile = File(...), target_column: str = Form(...)):
    try:
        # Sauvegarde du fichier temporaire pour l'entraînement
        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Appel de la fonction d'entraînement du modèle
        train_xgboost_model(file_path, target_column)
        logging.info("Entraînement du modèle terminé avec succès.")

        return {"status": "success", "message": "Entraînement du modèle XGBoost terminé avec succès."}
    except Exception as e:
        logging.error(f"Erreur pendant l'entraînement : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur pendant l'entraînement du modèle : {e}")
    finally:
        # Suppression du fichier temporaire après l'entraînement
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Fichier supprimé : {file_path}")


############# métriques de l'entraînement #############


@app.get("/metrics")
def get_metrics():
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("Your Experiment Name Here")
        if experiment is None:
            return {"error": "Experiment not found"}

        latest_run = client.search_runs(
            experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1
        )
        if latest_run:
            run = latest_run[0]
            metrics = run.data.metrics
            return {"metrics": metrics}
        else:
            return {"error": "No runs found"}
    except Exception as e:
        logging.error(f"Erreur pendant la récupération des métriques : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur pendant la récupération des métriques : {e}")


############# lien de la dernière exécution MLflow #############


@app.get("/last_mlflow_run_link")
def get_last_mlflow_run_link():
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("Your Experiment Name Here")
        if experiment is None:
            return {"error": "Experiment not found"}

        latest_run = client.search_runs(
            experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1
        )
        if latest_run:
            run_id = latest_run[0].info.run_id
            experiment_id = experiment.experiment_id
            # Générer le lien avec le bon port
            mlflow_run_link = f"http://localhost:5001/#/experiments/{experiment_id}/runs/{run_id}"
            return {"mlflow_run_link": mlflow_run_link}
        else:
            return {"error": "No runs found"}
    except Exception as e:
        logging.error(f"Erreur lors de la récupération du lien MLflow : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération du lien MLflow : {e}")



################################################################################################################

################# Extraction de données #################

# Endpoint pour extraire des données d'un fichier téléchargé
@app.post("/extract_file_data")
async def extract_file_endpoint(file: UploadFile = File(...)):
    try:
        file_path = extract_file_data(file, DATA_DIR)
        logging.info(f"Fichier extrait et sauvegardé à : {file_path}")
        return {"status": "success", "file_path": file_path}
    except Exception as e:
        logging.error(f"Erreur lors de l'extraction du fichier : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'extraction du fichier : {e}")


# Endpoint pour extraire des données depuis S3
@app.post("/extract_s3_data")
async def extract_s3_endpoint(bucket_name: str = Form(...), file_name: str = Form(...)):
    try:
        file_path = extract_s3_data(bucket_name, file_name)
        logging.info(f"Fichier S3 extrait et sauvegardé à : {file_path}")
        return {"status": "success", "file_path": file_path}
    except Exception as e:
        logging.error(f"Erreur lors de l'extraction depuis S3 : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'extraction depuis S3 : {e}")


# Endpoint pour extraire des données depuis une base de données PostgreSQL


@app.post("/extract_db_data")
async def extract_db_data():
    query = "SELECT * FROM users"
    df = pd.read_sql(query, engine)
    file_path = os.path.join(DATA_DIR, "db_data.csv")
    df.to_csv(file_path, index=False)
    return {"status": "Data extracted from database", "file_path": file_path}


################################################################################################################

################# Traitement de données #################

# Endpoint pour le traitement de fichiers multiples
@app.post("/process_multiple_files")
async def process_multiple_files(files: List[UploadFile] = File(...), merge_key: Optional[str] = None):
    session_dir = os.path.join(CLEAN_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)
    file_paths = []

    # Sauvegarde des fichiers téléchargés
    for file in files:
        file_path = os.path.join(session_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        file_paths.append(file_path)

    try:
        # Appel au prétraitement et fusion des fichiers
        merged_df = preprocess_files(file_paths, merge_key)
        merged_file_path = save_data(merged_df, session_dir, "merged_data.csv")

        # Appel à l'agrégation des données
        aggregated_df = aggregate_data(merged_df, group_key=merge_key)
        aggregated_file_path = save_data(aggregated_df, session_dir, "aggregated_data.csv")

        return {
            "status": "Processus complet réussi",
            "merged_file": merged_file_path,
            "aggregated_file": aggregated_file_path
        }
    except Exception as e:
        logging.error(f"Erreur pendant le processus : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur pendant le processus : {e}")


################################################################################################################

################# Insertion de données #################

# Endpoint pour l'insertion des données dans PostgreSQL
@app.post("/upload_to_postgres")
async def upload_to_postgres(file: UploadFile = File(...), table_name: str = Form(None)):
    global file_path
    try:
        # Si table_name n'est pas fourni, utiliser le nom du fichier nettoyé comme nom de table
        if not table_name:
            table_name = clean_table_name(file.filename)

        # Sauvegarde du fichier pour traitement
        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Appel à la fonction d'insertion dans la base de données
        add_to_postgres(file_path, table_name)
        logging.info(f"Les données ont été insérées dans la table '{table_name}' de PostgreSQL.")

        return {"status": "success", "message": f"Données insérées dans la table '{table_name}' avec succès."}
    except Exception as e:
        logging.error(f"Erreur pendant l'insertion en base de données : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur pendant l'insertion en base de données : {e}")
    finally:
        # Suppression du fichier temporaire après l'insertion
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Fichier supprimé : {file_path}")


########### Endpoint pour l'upload d'un fichier dans PostgreSQL############
@app.post("/upload_to_postgres")
async def upload_to_postgres(file: UploadFile = File(...), table_name: str = None):
    # Define the temporary path to save the file
    upload_directory = "data/uploaded_files"
    os.makedirs(upload_directory, exist_ok=True)

    # Save the file temporarily
    file_path = os.path.join(upload_directory, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # If table_name is not provided, generate it from the file name
    if table_name is None:
        table_name = clean_table_name(file.filename)

    try:
        # Insert data into PostgreSQL
        add_to_postgres(file_path, table_name)

        # Remove the temporary file after insertion
        os.remove(file_path)

        return {"status": "success", "message": f"Données insérées dans la table '{table_name}' avec succès."}
    except Exception as e:
        logging.error(f"Erreur pendant l'insertion dans PostgreSQL : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur d'insertion dans PostgreSQL : {e}")
