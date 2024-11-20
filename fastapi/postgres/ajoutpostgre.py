import shutil
import re

import pandas as pd
from fastapi import UploadFile, File
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration de la connexion PostgreSQL
DB_USER = os.environ['POSTGRES_USER']
DB_PASSWORD = os.environ['POSTGRES_PASSWORD']
DB_HOST = "postgres"
DB_PORT = "5432"
DB_NAME = os.environ['POSTGRES_DB']

# Créer l'engine SQLAlchemy pour PostgreSQL
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")


# Fonction de nettoyage pour transformer le nom de fichier en nom de table
def clean_table_name(filename: str) -> str:
    # Enlever l'extension du fichier
    base_name = os.path.splitext(filename)[0]
    # Remplacer les espaces et caractères spéciaux par des underscores
    clean_name = re.sub(r'\W+', '_', base_name)
    return clean_name.lower()  # On met en minuscule pour une cohérence des noms de tables


def add_to_postgres(file_path: str, table_name: str):
    """Ajoute les données du fichier CSV dans une table PostgreSQL."""
    # Charger le fichier CSV dans un DataFrame
    user_df = pd.read_csv(file_path)  # Remplacez par le chemin vers votre fichier CSV

    # Insérer les données du DataFrame dans PostgreSQL
    user_df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Les données du fichier CSV '{file_path}' ont été insérées dans la table '{table_name}' de PostgreSQL.")

    # Afficher les 5 premières lignes de la table PostgreSQL pour vérification
    print(pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5", engine))
