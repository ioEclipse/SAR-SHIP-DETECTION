FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Moins de dépendances système
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier d'abord requirements.txt seul (pour mieux utiliser le cache)
COPY requirements.txt .

# Installer les dépendances Python d'abord
RUN pip3 install --no-cache-dir -r requirements.txt

# Puis copier le reste
COPY app/ .

# Variables d'environnement
ENV INPUT_DIR=/input
ENV OUTPUT_DIR=/output
ENV MODEL_PATH=/app/best1.onnx

CMD ["python3", "Final_function.py"]