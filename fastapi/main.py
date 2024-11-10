from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
import os
import time
import subprocess
import requests
import pandas as pd
import json

from great_expectations.core.batch_spec import RuntimeDataBatchSpec
from great_expectations.data_context.types.base import DataContextConfig
from sqlalchemy import create_engine
import mlflow
from mlflow.exceptions import MlflowException
import great_expectations as ge
from great_expectations.data_context.data_context.file_data_context import FileDataContext
from typing import Dict, Any

app = FastAPI()
DATA_DIR = "data/datajson"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("data/dataclean", exist_ok=True)


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


@app.post("/train")
async def train_model(background_tasks: BackgroundTasks) -> Dict[str, str]:
    background_tasks.add_task(run_training)
    return {"status": "Training started"}


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

