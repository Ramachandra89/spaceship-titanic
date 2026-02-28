FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY app.py .

# Copy pre-trained model artefact (built during CI/CD or mounted as volume)
COPY outputs/models/ ./outputs/models/

# (Optional) Mount the MLflow SQLite db if running offline
# COPY mlflow.db ./mlflow.db

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
