# ARG to allow expert base image later
ARG BASE_IMAGE=python:3.11-slim

FROM ${BASE_IMAGE}

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
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]