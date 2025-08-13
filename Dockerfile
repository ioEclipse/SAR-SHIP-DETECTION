# Base stage
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the app
COPY FullApp/ ./FullApp/
COPY preprocessing/ ./preprocessing/
COPY tracking/ ./tracking/
COPY config.json ./config.json

EXPOSE 8501
WORKDIR /app/FullApp
CMD ["streamlit", "run", "home.py", "--server.port=8501", "--server.address=0.0.0.0"]
