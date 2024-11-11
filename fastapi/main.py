import logging
import shutil
import os
import time
import subprocess

import boto3
import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Dict, Any

from pydantic import BaseModel
from sqlalchemy import create_engine

from training.train_xgboost_mlflow import train_xgboost_model
import os

app = FastAPI()
DATA_DIR = "data/datajson"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("data/dataclean", exist_ok=True)
CLEAN_DIR = "data/dataclean"

UPLOAD_DIRECTORY = "uploaded_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

logging.basicConfig(level=logging.INFO)

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


class DataSource(BaseModel):
    source_type: str
    source_path: str


class ModelInput(BaseModel):
    data_path: str  # Exemple de paramètre d'entrée pour le chemin du fichier


def connect_to_mlflow() -> None:
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    for _ in range(5):
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            client = mlflow.tracking.MlflowClient()
            client.search_experiments()
            print("Connected to MLflow server.")
            return
        except MlflowException as e:
            print(f"Failed to connect to MLflow: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    raise RuntimeError("Could not connect to MLflow after multiple attempts.")


connect_to_mlflow()


@app.post("/train_xgboost_model")
async def train_model(file: UploadFile = File(...)):
    # Définir un chemin absolu pour le fichier temporaire
    file_path = os.path.abspath(os.path.join(UPLOAD_DIRECTORY, file.filename))

    try:
        logging.info("Début du téléversement du fichier...")

        # Sauvegarder le fichier téléversé localement
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Vérifier l'existence du fichier après la sauvegarde
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="Le fichier n'a pas été correctement sauvegardé.")

        logging.info(f"Fichier sauvegardé localement à {file_path}. Démarrage de l'entraînement du modèle...")

        # Appeler la fonction d’entraînement avec le chemin du fichier sauvegardé
        train_xgboost_model(file_path)

        logging.info("Entraînement terminé avec succès.")

        return {"status": "success", "message": "Entraînement du modèle XGBoost démarré avec succès."}
    except Exception as e:
        logging.error(f"Erreur pendant l'entraînement : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur pendant l'entraînement du modèle : {e}")
    finally:
        # Nettoyage : supprimer le fichier temporaire après l’entraînement
        if os.path.exists(file_path):
            os.remove(file_path)


def run_training() -> None:
    subprocess.run(["python", "/app/fastapi/train_xgboost_mlflow.py"])


@app.get("/metrics")
def get_metrics() -> Dict[str, Any]:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Sleep Health XGBoost Experiment")
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


@app.post("/extract_file_data")
async def extract_file_data(file: UploadFile = File(...)):
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "File uploaded successfully", "file_path": file_path}


@app.post("/extract_s3_data")
async def extract_s3_data(bucket_name: str, file_name: str):
    obj = s3_client.get_object(Bucket=bucket_name, Key=file_name)
    df = pd.read_csv(obj['Body'])
    file_path = os.path.join(DATA_DIR, "s3_data.csv")
    df.to_csv(file_path, index=False)
    return {"status": "Data extracted from S3", "file_path": file_path}


@app.post("/extract_db_data")
async def extract_db_data():
    try:
        logging.info("Starting database extraction.")
        query = "SELECT * FROM users"
        df = pd.read_sql(query, engine)
        logging.info("Data extracted from the database.")

        file_path = os.path.join(DATA_DIR, "db_data.csv")
        df.to_csv(file_path, index=False)
        logging.info(f"Data saved to {file_path}.")

        return {"status": "Data extracted from database", "file_path": file_path}
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return {"status": "Error", "message": str(e)}


@app.post("/process_all")
async def process_all():
    try:
        # Chemins des fichiers sources
        user_csv_path = os.path.join(DATA_DIR, "db_data.csv")
        transaction_csv_path = os.path.join(DATA_DIR, "s3_data.csv")
        product_csv_path = os.path.join(DATA_DIR, "products.csv")

        # Vérifier que tous les fichiers existent
        if not (os.path.exists(user_csv_path) and os.path.exists(transaction_csv_path) and os.path.exists(
                product_csv_path)):
            raise HTTPException(status_code=404, detail="Un ou plusieurs fichiers sources sont manquants.")

        # Étape 1 : Fusion des données
        user_df = pd.read_csv(user_csv_path)
        transaction_df = pd.read_csv(transaction_csv_path)
        product_df = pd.read_csv(product_csv_path)

        # Fusionner les DataFrames (en utilisant 'user_id' comme exemple de clé de fusion)
        merged_df = user_df.merge(transaction_df, on="user_id", how="left").merge(product_df, on="user_id", how="left")

        # Sauvegarder le fichier fusionné
        merged_file_path = os.path.join(CLEAN_DIR, "merged_data.csv")
        merged_df.to_csv(merged_file_path, index=False)
        logging.info(f"Données fusionnées sauvegardées à {merged_file_path}")

        # Étape 2 : Nettoyage des données
        # Exemple de nettoyage
        cleaned_df = merged_df.dropna()  # Supprimer les lignes avec des valeurs manquantes
        if "date_column" in cleaned_df.columns:
            cleaned_df["date_column"] = pd.to_datetime(cleaned_df["date_column"], errors="coerce")
            cleaned_df.dropna(subset=["date_column"], inplace=True)  # Supprimer les dates non valides

        # Sauvegarder le fichier nettoyé
        cleaned_file_path = os.path.join(CLEAN_DIR, "cleaned_merged_data.csv")
        cleaned_df.to_csv(cleaned_file_path, index=False)
        logging.info(f"Données nettoyées sauvegardées à {cleaned_file_path}")

        # Étape 3 : Validation des données
        # Ici, une validation simple pour s’assurer que certaines colonnes nécessaires existent
        if "user_id" not in cleaned_df.columns:
            raise HTTPException(status_code=422, detail="Validation échouée : 'user_id' est manquant.")

        # Étape 4 : Agrégation des données
        # Par exemple, regroupement par 'user_id' et somme des montants
        aggregated_df = cleaned_df.groupby("user_id").sum()

        # Sauvegarder le fichier agrégé
        aggregated_file_path = os.path.join(CLEAN_DIR, "aggregated_data.csv")
        aggregated_df.to_csv(aggregated_file_path)
        logging.info(f"Données agrégées sauvegardées à {aggregated_file_path}")

        # Retourner les chemins des fichiers générés
        return {
            "status": "Processus complet réussi",
            "merged_file": merged_file_path,
            "cleaned_file": cleaned_file_path,
            "aggregated_file": aggregated_file_path
        }

    except Exception as e:
        logging.error(f"Erreur pendant le processus : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur pendant le processus : {e}")



















































































































































# @app.post("/extract_api_data")
# async def extract_api_data(background_tasks: BackgroundTasks) -> Dict[str, str]:
#     background_tasks.add_task(fetch_api_data)
#     return {"status": "API data extraction started"}
#
#
# def fetch_api_data() -> None:
#     url = "https://api.example.com/data"
#     headers = {"Authorization": "Bearer YOUR_API_TOKEN"}
#     params = {"param1": "value1"}
#
#     response = requests.get(url, headers=headers, params=params)
#     if response.status_code == 200:
#         data = response.json()
#         with open("data/api_data.json", "w") as f:
#             json.dump(data, f)
#     else:
#         print("Failed to fetch API data:", response.status_code)
#
#
# @app.post("/extract_file_data")
# async def extract_file_data(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> Dict[str, str]:
#     file_location = f"data/{file.filename}"
#
#     with open(file_location, "wb") as f:
#         f.write(await file.read())
#
#     background_tasks.add_task(process_file_data, file_location)
#
#     return {"status": "File uploaded and processing started"}
#
#
# def process_file_data(file_location: str) -> None:
#     df = pd.read_csv(file_location)
#     output_path = f"data/datajson/{os.path.splitext(os.path.basename(file_location))[0]}.json"
#     df.to_json(output_path, orient="records")
#     print(f"Data saved to {output_path}")
#
#
# @app.post("/extract_db_data")
# async def extract_db_data(background_tasks: BackgroundTasks) -> Dict[str, str]:
#     background_tasks.add_task(fetch_db_data)
#     return {"status": "Database data extraction started"}
#
#
# def fetch_db_data() -> None:
#     engine = create_engine("postgresql://user:password@localhost/dbname")
#     query = "SELECT * FROM your_table"
#     df = pd.read_sql(query, engine)
#     df.to_csv("data/db_data.csv", index=False)
#
#
# @app.post("/run_sql_query")
# async def run_sql_query(background_tasks: BackgroundTasks) -> Dict[str, str]:
#     background_tasks.add_task(execute_sql_query)
#     return {"status": "SQL query execution started"}
#
#
# def execute_sql_query() -> None:
#     engine = create_engine("postgresql://user:password@localhost/dbname")
#     query = "SELECT user_id, AVG(score) FROM user_scores GROUP BY user_id"
#     df = pd.read_sql(query, engine)
#     df.to_csv("data/aggregated_data.csv", index=False)
#

# @app.post("/clean_data")
# async def clean_data(background_tasks: BackgroundTasks, file: UploadFile):
#     input_path = f"{DATA_DIR}/input_data.csv"
#     cleaned_path = f"{DATA_DIR}/cleaned_data.json"
#
#     # Save uploaded file
#     with open(input_path, "wb") as f:
#         f.write(file.file.read())
#
#     # Load DataFrame
#     try:
#         df = pd.read_csv(input_path)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")
#
#     # 1. Handling Missing Values with Column Checks
#     if "numeric_column" in df.columns:
#         df["numeric_column"].fillna(df["numeric_column"].mean(), inplace=True)
#     else:
#         print("Warning: 'numeric_column' not found in the CSV file.")
#
#     if "category_column" in df.columns:
#         df["category_column"].fillna("Unknown", inplace=True)
#     else:
#         print("Warning: 'category_column' not found in the CSV file.")
#
#     if "date_column" in df.columns:
#         df["date_column"] = pd.to_datetime(df["date_column"], errors="coerce").fillna(pd.to_datetime("1900-01-01"))
#     else:
#         print("Warning: 'date_column' not found in the CSV file.")
#
#     # 2. Outlier Detection and Handling
#     if "numeric_column" in df.columns:
#         q1 = df["numeric_column"].quantile(0.25)
#         q3 = df["numeric_column"].quantile(0.75)
#         iqr = q3 - q1
#         lower_bound = q1 - 1.5 * iqr
#         upper_bound = q3 + 1.5 * iqr
#         df["numeric_column"] = df["numeric_column"].clip(lower_bound, upper_bound)
#     else:
#         print("Warning: 'numeric_column' not found in the CSV file, skipping outlier handling.")
#
#     # 3. Standardization/Normalization
#     if "numeric_column" in df.columns:
#         min_val = df["numeric_column"].min()
#         max_val = df["numeric_column"].max()
#         if min_val != max_val:  # Avoid division by zero
#             df["numeric_column"] = (df["numeric_column"] - min_val) / (max_val - min_val)
#         else:
#             df["numeric_column"] = 0.5  # Assign a constant if all values are the same
#     else:
#         print("Warning: 'numeric_column' not found in the CSV file, skipping normalization.")
#
#     # 4. Data Type Conversion
#     if "category_column" in df.columns:
#         df["category_column"] = df["category_column"].astype("category")
#     else:
#         print("Warning: 'category_column' not found in the CSV file, skipping type conversion.")
#
#     # 5. Text Data Handling
#     if "text_column" in df.columns:
#         df["text_column"] = df["text_column"].str.strip().str.lower()
#     else:
#         print("Warning: 'text_column' not found in the CSV file, skipping text handling.")
#
#     # Save cleaned data
#     df.to_json(cleaned_path, orient="records")
#
#     return {"status": "Data cleaned", "cleaned_data_path": cleaned_path}
#
#
# class BaseDataContext:
#     pass
#
#
# @app.post("/validate_data")
# async def validate_data(background_tasks: BackgroundTasks):
#     # Path to the cleaned data
#     cleaned_data_path = f"{DATA_DIR}/cleaned_data.json"
#
#     # Load the data
#     try:
#         df = pd.read_json(cleaned_data_path)
#     except ValueError as e:
#         raise HTTPException(status_code=500, detail=f"Could not load cleaned data: {e}")
#
#     # Convert the DataFrame to a Great Expectations dataset
#     ge_df = PandasDataset(df)
#
#     # Example Expectations (adapt based on your actual data structure)
#     # Check for no null values in specific columns
#     validation_results = {
#         "no_null_values": ge_df.expect_column_values_to_not_be_null("your_column_name").success if "your_column_name" in ge_df.columns else None,
#         "unique_values": ge_df.expect_column_values_to_be_unique("your_column_name").success if "your_column_name" in ge_df.columns else None,
#         "numeric_range": ge_df.expect_column_values_to_be_between("numeric_column", min_value=0, max_value=100).success if "numeric_column" in ge_df.columns else None
#     }
#
#     # Determine if all validations passed
#     all_checks_passed = all(result for result in validation_results.values() if result is not None)
#
#     return {"validation_results": validation_results, "all_checks_passed": all_checks_passed}
#
#
# @app.post("/aggregate_data")
# async def aggregate_data():
#     cleaned_path = f"{DATA_DIR}/cleaned_data.json"
#     aggregated_path = f"{DATA_DIR}/aggregated_data.csv"
#
#     if not os.path.exists(cleaned_path):
#         return {"error": "Cleaned data not found. Please run /clean_data first."}
#
#     # Load cleaned data
#     df = pd.read_json(cleaned_path)
#
#     # Identify potential grouping and numeric columns dynamically
#     group_columns = [col for col in df.columns if df[col].dtype == 'object' or df[col].nunique() < 50]
#     numeric_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
#
#     # Ensure we have at least one grouping column and one numeric column
#     if not group_columns or not numeric_columns:
#         return {"error": "Suitable columns for aggregation not found in data."}
#
#     # Select the first detected group column and aggregate numeric columns
#     group_column = group_columns[0]
#     aggregation_methods = {num_col: 'sum' if 'count' not in num_col.lower() else 'mean' for num_col in numeric_columns}
#
#     # Perform aggregation
#     aggregated_df = df.groupby(group_column).agg(aggregation_methods)
#
#     # Save aggregated data
#     aggregated_df.to_csv(aggregated_path)
#
#     return {"status": "Data aggregated", "aggregated_data_path": aggregated_path}

