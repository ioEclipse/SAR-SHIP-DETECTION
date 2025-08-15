# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# NVIDIA Jetson AGX Orin Dockerfile for SAR Ship Detection
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Add fonts for YOLO
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

# Install CUDA keyring and system dependencies
ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb .
RUN dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    libopenmpi-dev \
    libopenblas-base \
    libomp-dev \
    libcusparselt0 \
    libcusparselt-dev \
    # Additional dependencies for SAR processing
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libspatialindex-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb

WORKDIR /app

# Copy only Jetson AGX Orin directory contents
COPY . .

# Install NVIDIA optimized packages for Jetson
RUN python3 -m pip install --upgrade pip uv && \
    uv pip install --system \
    # Core NVIDIA optimized packages
    numpy==1.26.4 \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl && \
    # Install Ultralytics with export capabilities
    uv pip install --system ultralytics[export] && \
    # Install CUDA-optimized packages for SAR processing
    uv pip install --system \
    cupy-cuda12x \
    opencv-python-headless \
    Pillow \
    rasterio \
    && rm -rf *.whl /root/.config/Ultralytics/persistent_cache.json

# Set default command to run inference
CMD ["python3", "ourmodel_inference_function.py"]