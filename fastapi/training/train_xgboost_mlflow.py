import logging

import xgboost as xgb
import mlflow
import mlflow.xgboost
from fastapi import HTTPException
from openpyxl.reader.excel import load_workbook
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import os


def train_xgboost_model(data_path: str):
    # Vérifier l'existence du fichier
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Le fichier {data_path} est introuvable.")

    logging.info(f"Chargement du fichier à partir du chemin : {data_path}")

    # Chargement et diagnostic du fichier
    try:
        with open(data_path, "rb") as f:
            content = f.read(1024)  # Lire les 1024 premiers octets
        logging.info(f"Le fichier est accessible et les premiers octets sont lus avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors de la lecture binaire du fichier : {e}")
        raise ValueError(f"Erreur lors de la lecture binaire du fichier : {e}")

    # Lire avec pandas après validation de l'accès binaire
    try:
        if data_path.endswith('.xlsx'):
            logging.info("Tentative de chargement du fichier avec pandas (read_excel).")
            df = pd.read_excel(data_path)
        elif data_path.endswith('.csv'):
            logging.info("Tentative de chargement du fichier avec pandas (read_csv).")
            df = pd.read_csv(data_path)
        else:
            raise ValueError("Le fichier doit être au format .csv ou .xlsx")

        logging.info(f"Le fichier a été chargé avec succès. Aperçu :\n{df.head()}")
    except Exception as e:
        logging.error(f"Erreur pendant le chargement du fichier dans pandas : {e}")
        raise ValueError(f"Erreur pendant le chargement du fichier dans pandas : {e}")

    # Pré-traitement
    df.drop(columns=['Person ID'], inplace=True, errors='ignore')
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('Good Sleep')
    df.dropna(thresh=int(0.8 * len(df)), axis=1, inplace=True)
    df['BMI Category'].replace(
        {"Norm": "Normal Weight", "Norma": "Normal Weight", "Normal": "Normal Weight", "Nan": "Normal Weight"},
        inplace=True)

    # Encoder les caractéristiques catégorielles
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Définir les caractéristiques et la cible
    X = df.drop(columns=['Sleep Disorder'])
    y = df['Sleep Disorder']

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configuration de l'expérience MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Sleep Health XGBoost Experiment")

    with mlflow.start_run():
        # Entraîner le modèle XGBoost
        xgb_model = xgb.XGBClassifier(max_depth=3, n_estimators=40, use_label_encoder=False, eval_metric='mlogloss')
        xgb_model.fit(X_train, y_train)

        # Faire des prédictions et évaluer
        y_pred = xgb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Enregistrer les paramètres et métriques dans MLflow
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("n_estimators", 50)
        mlflow.log_metric("accuracy", accuracy)

        # Enregistrer le modèle
        mlflow.xgboost.log_model(xgb_model, "model")

    print("Entraînement du modèle terminé avec suivi MLflow.")
