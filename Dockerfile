FROM python:3.12.14-trixie

WORKDIR /prod

# Librairies système nécessaires à OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Code de l'application
COPY app /prod/app

# Requirements
COPY requirements-cloud.txt /prod/requirements-cloud.txt
COPY requirements-api.txt /prod/requirements-api.txt

# Mise à jour de pip
RUN pip install --no-cache-dir --upgrade pip

# PyTorch CPU uniquement pour YOLO / Ultralytics
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Dépendances ML pour Cloud Run CPU
RUN pip install --no-cache-dir -r /prod/requirements-cloud.txt

# Dépendances FastAPI
RUN pip install --no-cache-dir -r /prod/requirements-api.txt

# Lancement de l'API
CMD ["sh", "-c", "uvicorn app.api.fast:app --host 0.0.0.0 --port ${PORT:-8000}"]
