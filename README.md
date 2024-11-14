"""
# Projet de Traitement et Entraînement de Modèle ML avec XGBoost

Bienvenue dans ce projet qui propose un système complet pour l'extraction, le prétraitement, l'entraînement et la surveillance de modèles de machine learning, en utilisant principalement **XGBoost** pour la classification, **MLflow** pour le suivi des expérimentations, et **Streamlit** pour une interface utilisateur interactive.

## Table des Matières
- [Aperçu du Projet](#aperçu-du-projet)
- [Fonctionnalités](#fonctionnalités)
- [Architecture du Projet](#architecture-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
  - [Exécution du Serveur API](#exécution-du-serveur-api)
  - [Utilisation de l'Interface Streamlit](#utilisation-de-linterface-streamlit)
- [Endpoints Principaux de l'API](#endpoints-principaux-de-lapi)
- [Exemples d'Utilisation](#exemples-dutilisation)
- [Structure des Dossiers](#structure-des-dossiers)
- [Contributions](#contributions)

## Aperçu du Projet

Ce projet fournit une solution intégrée pour entraîner des modèles de machine learning avec XGBoost, en utilisant des fichiers de données provenant de diverses sources (téléchargement de fichiers, bases de données PostgreSQL, et stockage S3). L'ensemble du workflow est géré par une API FastAPI, avec des métriques suivies via MLflow. Les utilisateurs peuvent interagir avec le système et visualiser les métriques d'entraînement grâce à une interface Streamlit.

## Fonctionnalités

- **Extraction de Données** : Extraction de données depuis des fichiers, des bases de données PostgreSQL et des sources S3.
- **Prétraitement des Données** : Nettoyage des valeurs manquantes et des valeurs aberrantes.
- **Entraînement de Modèle XGBoost** : Entraînement d'un modèle avec la possibilité de réutiliser un modèle sauvegardé.
- **Suivi des Expérimentations** : Intégration avec MLflow pour le suivi des métriques et des paramètres d'entraînement.
- **Visualisation des Métriques** : Affichage des métriques et des liens d'exécution MLflow via une interface Streamlit.
- **Surveillance avec Prometheus et Grafana** (optionnel) : Pour surveiller les performances et la stabilité du système.

## Architecture du Projet

Le projet est structuré en trois parties principales :
1. **API FastAPI** - Gère le backend pour l'extraction de données, l'entraînement de modèles, et le suivi des métriques.
2. **Interface Streamlit** - Fournit une interface utilisateur pour interagir avec l'API, lancer les entraînements, et visualiser les métriques.
3. **Suivi avec MLflow** - Gère le suivi des expérimentations et le stockage des modèles.

## Prérequis

Avant de démarrer, assurez-vous d'avoir les éléments suivants installés :
- **Docker** (pour la gestion des services de base de données et de stockage S3)
- **Python 3.7+**
- **PostgreSQL** (si vous utilisez une base de données externe pour le stockage)
- **MinIO** ou **AWS S3** (pour le stockage des artefacts MLflow)

## Installation

1. **Cloner le projet** :
   ```bash
   git clone https://github.com/votre-repo/votre-projet.git
   cd votre-projet
Configurer l'environnement :

Créez un fichier .env à la racine du projet pour y définir les variables d'environnement nécessaires (ex: POSTGRES_USER, POSTGRES_PASSWORD, MINIO_ACCESS_KEY, etc.)
Installer les dépendances :

bash
Toujours afficher les détails

Copier le code
pip install -r requirements.txt
Démarrer les services Docker (optionnel) :

bash
Toujours afficher les détails

Copier le code
docker-compose up -d
Utilisation
Exécution du Serveur API
Lancer l'API FastAPI :

bash
Toujours afficher les détails

Copier le code
uvicorn main:app --host 0.0.0.0 --port 8000
Utilisation de l'Interface Streamlit
Pour lancer l'interface utilisateur Streamlit :

bash
Toujours afficher les détails

Copier le code
streamlit run streamlit_app.py
Endpoints Principaux de l'API
/extract_file_data : Extraire les données d'un fichier téléchargé.
/extract_s3_data : Extraire les données depuis un stockage S3.
/extract_db_data : Extraire les données depuis une base de données PostgreSQL.
/train_xgboost_model : Entraîner un modèle XGBoost avec des paramètres personnalisés.
/metrics : Récupérer les métriques d'entraînement du dernier modèle.
/last_mlflow_run_link : Récupérer le lien MLflow vers la dernière exécution du modèle.
Exemples d'Utilisation
Exemple d'Entraînement de Modèle
Pour entraîner un modèle, téléchargez un fichier et spécifiez la colonne cible :

python
Toujours afficher les détails

Copier le code
import requests

url = "http://localhost:8000/train_xgboost_model"
files = {"file": open("data/dataset.csv", "rb")}
data = {"target_column": "target", "experiment_name": "Experiment_1"}
response = requests.post(url, files=files, data=data)
print(response.json())
Exemple de Visualisation des Métriques sur Streamlit
Lancez Streamlit, puis accédez aux pages de visualisation pour voir les dernières métriques d'entraînement et le lien vers MLflow pour explorer plus en détail.

Structure des Dossiers
data/ - Dossier pour stocker les fichiers de données téléchargés.
saved_models/ - Dossier pour sauvegarder les modèles.
streamlit_app/ - Code de l'application Streamlit.
mlflow/ - Fichiers de configuration pour MLflow.
requirements.txt - Liste des dépendances Python.
Contributions
Les contributions sont les bienvenues ! N'hésitez pas à soumettre des issues ou des pull requests pour améliorer le projet.