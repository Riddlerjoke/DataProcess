FROM python:3.9

WORKDIR /app

# Copier et installer les dépendances
COPY ./fastapi/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copier le code source FastAPI dans le conteneur
COPY ./fastapi /app

# Commande pour démarrer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
