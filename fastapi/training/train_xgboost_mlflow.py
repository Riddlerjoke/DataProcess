import logging
from typing import Optional
import os
import xgboost as xgb
import mlflow
import mlflow.xgboost
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd


def train_xgboost_model(data_path: str, target_column: str = 'target', experiment_name: str = 'default_experiment',
                        save_local_path: Optional[str] = "saved_models/xgb_model.joblib"):
    # Vérifier l'existence du fichier
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Le fichier {data_path} est introuvable.")

    logging.info(f"Chargement du fichier à partir du chemin : {data_path}")

    # Chargement et traitement du fichier
    try:
        if data_path.endswith('.xlsx'):
            df = pd.read_excel(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path, engine='python', sep=r'[;,\t|]', na_values=['', ' ', 'NA', 'NaN'])
        else:
            raise ValueError("Format de fichier non pris en charge. Veuillez fournir un fichier .csv ou .xlsx.")

        # Supprimer les colonnes avec plus de 20 % de valeurs manquantes
        df = df.loc[:, df.isnull().mean() < 0.2]
        logging.info(f"Colonnes restantes après suppression : {df.columns.tolist()}")

        # Convertir toutes les colonnes mixtes en texte
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].apply(type).nunique() > 1:
                df[col] = df[col].astype(str)

        logging.info(f"Fichier chargé avec succès. Aperçu des données :\n{df.head()}")
    except Exception as e:
        logging.error(f"Erreur pendant le chargement du fichier : {e}")
        raise ValueError(f"Erreur pendant le chargement du fichier : {e}")

    # Vérifier la présence de la colonne cible
    if target_column not in df.columns:
        raise ValueError(f"La colonne cible '{target_column}' est manquante dans les données.")

    # Remplacer les valeurs manquantes dans la colonne cible
    df[target_column].fillna(df[target_column].mode()[0], inplace=True)

    # Encoder la colonne cible avec LabelEncoder
    target_encoder = LabelEncoder()
    try:
        df[target_column] = target_encoder.fit_transform(df[target_column].astype(str))
    except Exception as e:
        logging.error(f"Erreur pendant l'encodage de la colonne cible : {e}")
        raise ValueError(f"Erreur pendant l'encodage de la colonne cible : {e}")

    # Encoder les autres colonnes catégorielles
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        if column != target_column:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            label_encoders[column] = le

    # Définir les caractéristiques (X) et la cible (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Remplir les valeurs manquantes
    X.fillna(X.mode().iloc[0], inplace=True)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configurer MLflow et entraîner le modèle
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        xgb_model = xgb.XGBClassifier(max_depth=3, n_estimators=40, use_label_encoder=False, eval_metric='mlogloss')
        xgb_model.fit(X_train, y_train)

        # Faire des prédictions et évaluer
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)  # Probabilités pour ROC AUC

        # Ensure alignment of y_test and predicted probabilities
        if len(set(y_test)) != y_pred_proba.shape[1]:
            # Align the number of classes in y_test and y_pred_proba
            y_test_binarized = label_binarize(y_test, classes=range(y_pred_proba.shape[1]))
        else:
            y_test_binarized = label_binarize(y_test, classes=list(set(y_test)))

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        try:
            auc = roc_auc_score(y_test_binarized, y_pred_proba, multi_class="ovr")
        except Exception as e:
            logging.warning(f"Erreur dans le calcul de l'AUC : {e}")
            auc = None

        # Log metrics in MLflow
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("n_estimators", 40)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        if auc is not None:
            mlflow.log_metric("auc", auc)

        # Save and log the model
        if save_local_path:
            directory = os.path.dirname(save_local_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
                logging.info(f"Répertoire créé : {directory}")
            dump(xgb_model, save_local_path)
            mlflow.log_artifact(save_local_path, "model")
            logging.info(f"Modèle sauvegardé localement à : {save_local_path}")

        mlflow.xgboost.log_model(xgb_model, "model")