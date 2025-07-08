FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements_fastapi.txt ./

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install -r requirements.txt && \
    pip3 install -r requirements_fastapi.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p model_zoo/lam_models/releases/lam/lam-20k/step_045500/ && \
    mkdir -p model_zoo/flame_tracking_models/ && \
    mkdir -p assets/sample_motion/export/ && \
    mkdir -p assets/sample_oac/ && \
    mkdir -p output/tracking/ && \
    mkdir -p /tmp

# Set permissions
RUN chmod +x deployment/deploy_ubuntu.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python3", "app_fastapi.py"] 