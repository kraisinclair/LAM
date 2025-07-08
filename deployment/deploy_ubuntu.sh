#!/bin/bash

# LAM FastAPI Deployment Script for Ubuntu Server
# This script sets up the LAM FastAPI service on Ubuntu 20.04/22.04

set -e  # Exit on any error

# Configuration
PROJECT_DIR="/opt/LAM"
SERVICE_NAME="lam-api"
PYTHON_VERSION="3.10"
CUDA_VERSION="11.8"  # or 12.1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root"
        exit 1
    fi
}

install_system_dependencies() {
    log_info "Installing system dependencies..."
    
    sudo apt-get update
    sudo apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
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
        nginx \
        supervisor
}

install_cuda() {
    log_info "Installing CUDA ${CUDA_VERSION}..."
    
    # Check if CUDA is already installed
    if command -v nvcc &> /dev/null; then
        log_warn "CUDA appears to be already installed"
        nvcc --version
        return
    fi
    
    # Install CUDA 11.8
    if [[ "$CUDA_VERSION" == "11.8" ]]; then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
        sudo apt-get update
        sudo apt-get -y install cuda-11-8
    # Install CUDA 12.1
    elif [[ "$CUDA_VERSION" == "12.1" ]]; then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
        sudo apt-get update
        sudo apt-get -y install cuda-12-1
    fi
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
}

setup_project() {
    log_info "Setting up project directory..."
    
    # Create project directory
    sudo mkdir -p $PROJECT_DIR
    sudo chown -R $USER:$USER $PROJECT_DIR
    
    # Copy project files (assuming we're running from the project directory)
    cp -r . $PROJECT_DIR/
    cd $PROJECT_DIR
    
    # Create virtual environment
    python${PYTHON_VERSION} -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch with CUDA support
    if [[ "$CUDA_VERSION" == "11.8" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif [[ "$CUDA_VERSION" == "12.1" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi
    
    # Install project requirements
    pip install -r requirements.txt
    pip install -r requirements_fastapi.txt
    
    # Install additional dependencies for LAM
    bash scripts/install/install_cu${CUDA_VERSION//.}.sh
}

setup_models() {
    log_info "Setting up model directories..."
    
    mkdir -p model_zoo/lam_models/releases/lam/lam-20k/step_045500/
    mkdir -p model_zoo/flame_tracking_models/
    mkdir -p assets/sample_motion/export/
    mkdir -p assets/sample_oac/
    mkdir -p output/tracking/
    
    log_warn "Please download the required model files to:"
    log_warn "  - model_zoo/lam_models/releases/lam/lam-20k/step_045500/"
    log_warn "  - model_zoo/flame_tracking_models/"
    log_warn "  - assets/sample_motion/export/"
    log_warn "  - assets/sample_oac/"
}

setup_service() {
    log_info "Setting up systemd service..."
    
    # Update paths in service file
    sed -i "s|/opt/LAM|$PROJECT_DIR|g" deployment/lam-api.service
    sed -i "s|User=ubuntu|User=$USER|g" deployment/lam-api.service
    sed -i "s|Group=ubuntu|Group=$USER|g" deployment/lam-api.service
    
    # Install service file
    sudo cp deployment/lam-api.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable $SERVICE_NAME
    
    log_info "Service installed. Use the following commands to manage it:"
    log_info "  sudo systemctl start $SERVICE_NAME"
    log_info "  sudo systemctl stop $SERVICE_NAME"
    log_info "  sudo systemctl status $SERVICE_NAME"
    log_info "  sudo journalctl -u $SERVICE_NAME -f"
}

setup_nginx() {
    log_info "Setting up Nginx reverse proxy..."
    
    cat > /tmp/lam-api-nginx.conf << 'EOF'
server {
    listen 80;
    server_name _;
    
    client_max_body_size 100M;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeouts for long-running requests
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
EOF
    
    sudo mv /tmp/lam-api-nginx.conf /etc/nginx/sites-available/lam-api
    sudo ln -sf /etc/nginx/sites-available/lam-api /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    
    # Test nginx configuration
    sudo nginx -t
    sudo systemctl restart nginx
    sudo systemctl enable nginx
}

create_test_script() {
    log_info "Creating test script..."
    
    cat > test_api.py << 'EOF'
#!/usr/bin/env python3
import requests
import json

def test_lam_api():
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Health: {response.json()}")
    
    # Test motions endpoint
    print("\nTesting motions endpoint...")
    response = requests.get(f"{base_url}/motions")
    print(f"Motions: {response.json()}")
    
    # Test root endpoint
    print("\nTesting root endpoint...")
    response = requests.get(f"{base_url}/")
    print(f"Root: {response.json()}")
    
    print("\nAPI is accessible!")

if __name__ == "__main__":
    test_lam_api()
EOF
    
    chmod +x test_api.py
}

main() {
    log_info "Starting LAM FastAPI deployment..."
    
    check_root
    install_system_dependencies
    install_cuda
    setup_project
    setup_models
    setup_service
    setup_nginx
    create_test_script
    
    log_info "Deployment completed!"
    log_info ""
    log_info "Next steps:"
    log_info "1. Download and place the required model files in the model_zoo directory"
    log_info "2. Start the service: sudo systemctl start $SERVICE_NAME"
    log_info "3. Check service status: sudo systemctl status $SERVICE_NAME"
    log_info "4. Test the API: python3 test_api.py"
    log_info "5. Access the API documentation at: http://your-server-ip/docs"
    log_info ""
    log_info "Logs can be viewed with: sudo journalctl -u $SERVICE_NAME -f"
}

# Run main function
main "$@" 