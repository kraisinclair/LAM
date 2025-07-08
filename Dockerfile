FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"
ENV TCNN_CUDA_ARCHITECTURES="60;61;70;75;80;86"

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
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements_docker.txt requirements_fastapi.txt ./

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch first with CUDA 11.8 support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install FastAPI dependencies (no CUDA compilation needed)
RUN pip3 install -r requirements_fastapi.txt

# Install main dependencies from docker-specific requirements
RUN pip3 install --no-cache-dir -r requirements_docker.txt

# Try to install pre-built pytorch3d wheel, fallback to source build
RUN pip3 install --no-cache-dir \
    --extra-index-url https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html \
    pytorch3d || \
    (echo "Pre-built pytorch3d failed, trying source build..." && \
     MAX_JOBS=1 pip3 install --no-cache-dir \
     git+https://github.com/facebookresearch/pytorch3d.git@stable) || \
    echo "pytorch3d installation failed, continuing without it..."

# Install CUDA packages with proper environment and fallbacks
RUN TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX" \
    FORCE_CUDA=1 \
    MAX_JOBS=1 \
    CUDA_VISIBLE_DEVICES=0 \
    pip3 install --no-cache-dir --no-build-isolation \
    "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git" || \
    echo "WARNING: diff-gaussian-rasterization install failed - some features may not work"

RUN TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX" \
    FORCE_CUDA=1 \
    MAX_JOBS=1 \
    CUDA_VISIBLE_DEVICES=0 \
    pip3 install --no-cache-dir --no-build-isolation \
    "git+https://github.com/camenduru/simple-knn.git" || \
    echo "WARNING: simple-knn install failed - some features may not work"

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