# Utiliser une image Python
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt et installer les dépendances
COPY ./streamlitapp/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Installer uvicorn s'il n'est pas déjà dans requirements.txt
RUN pip install uvicorn

# Copier le code source dans le conteneur
COPY ./fastapi /app/src

# Démarrer l'application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
