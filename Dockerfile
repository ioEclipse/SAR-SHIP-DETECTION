# ARG to allow expert base image later
ARG BASE_IMAGE=python:3.11-slim

FROM ${BASE_IMAGE}

# Install system dependencies needed for headless OpenCV and geospatial libraries
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgdal-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy global dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source
COPY FullApp/ ./FullApp/
COPY preprocessing/ ./preprocessing/
COPY tracking/ ./tracking/
COPY config.json ./config.json

# Expose port for Streamlit
EXPOSE 8501

# Set working directory to FullApp for correct relative paths
WORKDIR /app/FullApp

# Default command
CMD ["streamlit", "run", "home.py", "--server.port=8501", "--server.address=0.0.0.0"]