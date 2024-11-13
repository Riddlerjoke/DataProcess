import logging

import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import os

separators = r'[;,|]\s*'


def train_xgboost_model(data_path: str, target_column: str):
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

        # Supprimer les colonnes avec plus de 20 % de données manquantes
        threshold = 0.2
        df = df.loc[:, df.isnull().mean() < threshold]
        logging.info(f"Colonnes avec plus de 20% de données manquantes supprimées. Colonnes restantes : {df.columns.tolist()}")

        # Convertir toutes les colonnes mixtes en texte
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].apply(type).nunique() > 1:
                df[col] = df[col].astype(str)

        logging.info(f"Fichier chargé avec succès. Aperçu des données :\n{df.head()}")
    except Exception as e:
        logging.error(f"Erreur pendant le chargement du fichier dans pandas : {e}")
        raise ValueError(f"Erreur pendant le chargement du fichier dans pandas : {e}")

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

    # Traitement des valeurs manquantes de X
    X.fillna(X.mode().iloc[0], inplace=True)

    # Traitement des valeurs aberrantes de X et y en utilisant la méthode IQR
    for column in X.select_dtypes(include=['int', 'float']).columns:
        Q1 = X[column].quantile(0.25)
        Q3 = X[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = (X[column] >= lower_bound) & (X[column] <= upper_bound)
        X = X[mask]
        y = y[mask]

    # Traitement des valeurs manquantes de la cible (y)
    y.fillna(y.mode()[0], inplace=True)

    # Vérifier que les dimensions de X et y sont cohérentes
    if X.shape[0] != y.shape[0]:
        raise ValueError("Incohérence entre le nombre d'échantillons dans X et y après le prétraitement.")

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configurer MLflow et entraîner le modèle
    mlflow.set_experiment("Your Experiment Name Here")
    with mlflow.start_run():
        xgb_model = xgb.XGBClassifier(max_depth=3, n_estimators=40, use_label_encoder=False, eval_metric='mlogloss')
        xgb_model.fit(X_train, y_train)

        # Faire des prédictions et évaluer
        y_pred = xgb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Enregistrer les paramètres et métriques dans MLflow
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("n_estimators", 40)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.xgboost.log_model(xgb_model, "model")

    logging.info("Entraînement du modèle terminé avec succès.")