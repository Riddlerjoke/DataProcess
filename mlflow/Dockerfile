FROM python:3.11

# Install python package
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt*

# Copier les fichiers de MLflow, si nécessaire (dépend de votre configuration)
COPY . /app

# Commande pour démarrer MLflow
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres/${POSTGRES_DB}", \
     "--default-artifact-root", "s3://mlflow"]