FROM python:3.10-slim

WORKDIR /app

# Copier requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code de l'API
COPY app.py .
COPY modele/ ./modele/

# Exposer le port
EXPOSE 8002

# Démarrer Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
