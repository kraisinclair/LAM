#!/bin/bash

# LAM Docker Build Script
# This script provides multiple build strategies for the LAM FastAPI service

set -e

# Configuration
IMAGE_NAME="lam-api"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check Docker and NVIDIA runtime
check_requirements() {
    log_step "Checking requirements..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed!"
        exit 1
    fi
    
    if ! docker info | grep -q "nvidia"; then
        log_warn "NVIDIA Docker runtime not detected. CUDA packages may fail to build."
        log_warn "Install nvidia-docker2 for better CUDA support during build."
    fi
    
    log_info "Requirements check passed"
}

# Build with full CUDA support
build_full() {
    log_step "Building with full CUDA support..."
    
    docker build \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --progress=plain \
        -t "${FULL_IMAGE_NAME}" \
        -f Dockerfile \
        .
}

# Build with minimal CUDA (fallback)
build_minimal() {
    log_step "Building with minimal CUDA support (fallback)..."
    
    # Create temporary minimal Dockerfile
    cat > Dockerfile.minimal << 'EOF'
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

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

WORKDIR /app

# Copy requirements
COPY requirements_docker.txt requirements_fastapi.txt ./

# Install Python dependencies (no CUDA compilation)
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install -r requirements_fastapi.txt && \
    pip3 install -r requirements_docker.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p model_zoo/lam_models/releases/lam/lam-20k/step_045500/ && \
    mkdir -p model_zoo/flame_tracking_models/ && \
    mkdir -p assets/sample_motion/export/ && \
    mkdir -p assets/sample_oac/ && \
    mkdir -p output/tracking/ && \
    mkdir -p /tmp

EXPOSE 8000

CMD ["python3", "app_fastapi.py"]
EOF

    docker build \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --progress=plain \
        -t "${FULL_IMAGE_NAME}-minimal" \
        -f Dockerfile.minimal \
        .
        
    # Tag minimal as latest if main build failed
    docker tag "${FULL_IMAGE_NAME}-minimal" "${FULL_IMAGE_NAME}"
    
    # Cleanup
    rm -f Dockerfile.minimal
}

# Test the built image
test_image() {
    log_step "Testing the built image..."
    
    # Start container in background
    CONTAINER_ID=$(docker run -d --rm \
        --gpus all \
        -p 8000:8000 \
        "${FULL_IMAGE_NAME}")
    
    # Wait for startup
    sleep 10
    
    # Test health endpoint
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        log_info "Image test passed!"
        docker stop "${CONTAINER_ID}"
        return 0
    else
        log_error "Image test failed!"
        docker stop "${CONTAINER_ID}"
        return 1
    fi
}

# Show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --full      Build with full CUDA support (default)"
    echo "  --minimal   Build with minimal CUDA support"
    echo "  --test      Test the built image"
    echo "  --help      Show this help message"
}

# Main build function
main() {
    local BUILD_TYPE="full"
    local RUN_TEST=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --full)
                BUILD_TYPE="full"
                shift
                ;;
            --minimal)
                BUILD_TYPE="minimal"
                shift
                ;;
            --test)
                RUN_TEST=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    log_info "Starting LAM Docker build..."
    log_info "Build type: ${BUILD_TYPE}"
    
    check_requirements
    
    if [[ "${BUILD_TYPE}" == "full" ]]; then
        if build_full; then
            log_info "Full build completed successfully!"
        else
            log_warn "Full build failed, trying minimal build..."
            if build_minimal; then
                log_info "Minimal build completed successfully!"
            else
                log_error "Both builds failed!"
                exit 1
            fi
        fi
    elif [[ "${BUILD_TYPE}" == "minimal" ]]; then
        if build_minimal; then
            log_info "Minimal build completed successfully!"
        else
            log_error "Minimal build failed!"
            exit 1
        fi
    fi
    
    if [[ "${RUN_TEST}" == "true" ]]; then
        test_image
    fi
    
    log_info "Build completed!"
    log_info "Image: ${FULL_IMAGE_NAME}"
    log_info ""
    log_info "To run the container:"
    log_info "  docker run --gpus all -p 8000:8000 ${FULL_IMAGE_NAME}"
    log_info ""
    log_info "To run with docker-compose:"
    log_info "  docker-compose up -d"
}

main "$@" 