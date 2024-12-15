import logging
import os
import shutil
from datetime import datetime
from io import StringIO

import boto3
import mlflow
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

from joblib import load
from pydantic import BaseModel
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker

from dataOperation.data_processing import preprocess_files, aggregate_data, save_data, preprocess_for_prediction, \
    load_and_clean_file
from dataOperation.data_extraction import extract_file_data, extract_s3_data, extract_db_data, scrape_web_page
from dataOperation.predict import load_model_from_s3, predict
from training.train_xgboost_mlflow import train_xgboost_model
from postgres.ajoutpostgre import add_to_postgres, clean_table_name
from prometheus_fastapi_instrumentator import Instrumentator

s3_client = boto3.client(
    "s3",
    endpoint_url="http://minio:9000",
    aws_access_key_id=os.getenv('MINIO_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('MINIO_SECRET_ACCESS_KEY')
)
# Créez l'application FastAPI
app = FastAPI()

# Activer l'instrumentation pour Prometheus avant le démarrage
instrumentator = Instrumentator().instrument(app).expose(app)

# Configuration du middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration des répertoires
DATA_DIR = "data/uploaded_files"
CLEAN_DIR = "data/cleaned_files"
MODEL_DIR = "data/saved_models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("temp", exist_ok=True)

logging.basicConfig(level=logging.INFO)

engine = create_engine(
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_DB')}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class DBExtractionRequest(BaseModel):
    query: str
    table_name: Optional[str] = "extracted_data"


@app.on_event("startup")
async def startup_event():
    logging.info("Application démarrée avec succès.")


################ l'entraînement XGBoost ################
@app.post("/train_xgboost_model")
async def train_model(file: UploadFile = File(...), target_column: str = Form(...),
                      model_path: Optional[str] = Form(None),
                      experiment_name: Optional[str] = Form("Default Experiment")):
    try:
        # Sauvegarde du fichier temporaire pour l'entraînement
        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        if model_path and os.path.exists(model_path):
            # Rechargement du modèle existant
            logging.info(f"Chargement du modèle depuis : {model_path}")
            xgb_model = load(model_path)
            logging.info("Modèle rechargé avec succès.")
            # Utilisez le modèle pour des prédictions ou affinez-le
        else:
            # Entraînement d'un nouveau modèle avec un nom d'expérimentation spécifique
            logging.info("Aucun modèle existant fourni, entraînement d'un nouveau modèle.")
            train_xgboost_model(file_path, target_column, experiment_name=experiment_name,
                                save_local_path="saved_models/xgb_model.joblib")
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


############# Informations sur le dernier modèle entraîné #############
@app.get("/last_model_info")
def get_last_model_info():
    session = SessionLocal()
    try:
        # Récupérer le dernier run en se basant sur `start_time`
        last_run = session.execute(
            "SELECT * FROM runs ORDER BY start_time DESC LIMIT 1"
        ).fetchone()

        if not last_run:
            raise HTTPException(status_code=404, detail="Aucun modèle trouvé dans la base de données.")

        # Extraire les métriques et paramètres associés à ce `run_uuid`
        run_uuid = last_run['run_uuid']
        experiment_id = last_run['experiment_id']

        params = session.execute(
            "SELECT key, value FROM params WHERE run_uuid = :run_uuid", {'run_uuid': run_uuid}
        ).fetchall()

        metrics = session.execute(
            "SELECT key, value FROM metrics WHERE run_uuid = :run_uuid", {'run_uuid': run_uuid}
        ).fetchall()

        # Lien vers l'exécution dans MLflow
        mlflow_run_link = f"http://localhost:5001/#/experiments/{experiment_id}/runs/{run_uuid}"

        # Préparer les informations sous forme de dictionnaire
        model_info = {
            "run_id": run_uuid,
            "experiment_id": experiment_id,
            "start_time": last_run['start_time'],
            "artifact_uri": last_run['artifact_uri'],
            "metrics": {metric['key']: metric['value'] for metric in metrics},
            "params": {param['key']: param['value'] for param in params},
            "mlflow_run_link": mlflow_run_link
        }
        return model_info

    except Exception as e:
        logging.error(f"Erreur pendant la récupération des informations du modèle : {e}")
        raise HTTPException(status_code=500, detail="Erreur pendant la récupération des informations du modèle.")
    finally:
        session.close()


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


@app.post("/predict")
async def predict_endpoint(
        bucket_name: str = Form(...),
        model_path: str = Form(...),
        file: UploadFile = File(...)
):
    try:
        # Sauvegarder le fichier localement
        file_path = f"temp/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Charger et prétraiter les données
        df = preprocess_for_prediction(file_path)

        # Définir les colonnes attendues
        expected_columns = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Physical Activity Level',
                            'Stress Level', 'BMI Category', 'Blood Pressure', 'Heart Rate', 'Daily Steps',
                            'Sleep Disorder']

        # Charger le modèle depuis S3
        model = load_model_from_s3(bucket_name, model_path)

        # Effectuer la prédiction
        predictions = predict(df, model, expected_columns)

        return {"status": "success", "predictions": predictions.tolist()}
    except Exception as e:
        logging.error(f"Erreur pendant la prédiction : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur pendant la prédiction : {e}")
    finally:
        # Supprimer le fichier temporaire si nécessaire
        if os.path.exists(file_path):
            os.remove(file_path)


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


# Endpoint pour scraper des données depuis une URL
@app.post("/scrape_web_data")
async def scrape_web_data(url: str = Form(...)):
    """
    Endpoint pour extraire des données depuis une page web.

    Args:
        url (str): URL de la page web à scraper.

    Returns:
        dict: Statut et chemin du fichier contenant les données extraites.
    """
    try:
        file_path = scrape_web_page(url)
        logging.info(f"Données scrapées et sauvegardées à : {file_path}")
        return {"status": "success", "file_path": file_path}
    except Exception as e:
        logging.error(f"Erreur lors du scraping de l'URL : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du scraping : {e}")



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
