# LAM FastAPI Deployment Guide

This guide provides comprehensive instructions for deploying the LAM (Large Avatar Model) FastAPI service on an Ubuntu server.

## Overview

The LAM FastAPI service provides a REST API for processing images with the Large Avatar Model to generate animatable Gaussian heads. It includes:

- **Image Upload**: Accept image files via HTTP
- **Motion Selection**: Choose from available motion templates
- **ZIP Generation**: Create ZIP files for Open Avatar Chat integration
- **Video Output**: Generate animated videos with audio
- **Background Processing**: Handle long-running inference tasks

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM (recommended: RTX 3080 or better)
- **RAM**: Minimum 16GB system RAM (recommended: 32GB+)
- **Storage**: At least 50GB free space for models and temporary files
- **CPU**: Multi-core processor (8+ cores recommended)

### Software Requirements

- **OS**: Ubuntu 20.04 LTS or 22.04 LTS
- **CUDA**: Version 11.8 or 12.1
- **Python**: 3.10+
- **Docker**: Optional, for containerized deployment

## Quick Start

### Option 1: Automated Deployment Script

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd LAM
   ```

2. **Run the deployment script**:

   ```bash
   chmod +x deployment/deploy_ubuntu.sh
   ./deployment/deploy_ubuntu.sh
   ```

3. **Download required models** (see [Model Setup](#model-setup))

4. **Start the service**:
   ```bash
   sudo systemctl start lam-api
   ```

### Option 2: Docker Deployment

1. **Install Docker and nvidia-docker2**:

   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Build and run with docker-compose**:
   ```bash
   docker-compose up -d
   ```

## Manual Installation

### 1. System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    python3.10 python3.10-dev python3.10-venv \
    python3-pip git curl wget build-essential cmake \
    pkg-config libgl1-mesa-glx libglib2.0-0 libsm6 \
    libxext6 libxrender-dev libgomp1 ffmpeg zip unzip \
    nginx supervisor
```

### 2. CUDA Installation

**For CUDA 11.8**:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-11-8
```

**For CUDA 12.1**:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-1
```

Add to `~/.bashrc`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 3. Project Setup

```bash
# Create project directory
sudo mkdir -p /opt/LAM
sudo chown -R $USER:$USER /opt/LAM
cp -r . /opt/LAM/
cd /opt/LAM

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt
pip install -r requirements_fastapi.txt
```

### 4. Model Setup

Create the required directory structure and download models:

```bash
mkdir -p model_zoo/lam_models/releases/lam/lam-20k/step_045500/
mkdir -p model_zoo/flame_tracking_models/
mkdir -p assets/sample_motion/export/
mkdir -p assets/sample_oac/
```

**Required Model Files**:

1. **LAM Model** (`model_zoo/lam_models/releases/lam/lam-20k/step_045500/`):

   - `model.safetensors` - Main LAM model weights

2. **FLAME Tracking Models** (`model_zoo/flame_tracking_models/`):

   - `68_keypoints_model.pkl` - Facial landmark detection
   - `vgghead/vgg_heads_l.trcd` - VGG head detector
   - `matting/stylematte_synth.pt` - Human matting model
   - `FaceBoxesV2.pth` - Face detection model

3. **Motion Templates** (`assets/sample_motion/export/`):

   - Each motion should have a folder with:
     - `flame_param/` - Directory with FLAME parameters
     - `{motion_name}.wav` - Audio file

4. **Open Avatar Chat Assets** (`assets/sample_oac/`):
   - `template_file.fbx` - FBX template
   - `animation.glb` - Animation file

### 5. Service Configuration

Install systemd service:

```bash
sudo cp deployment/lam-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable lam-api
```

Configure Nginx:

```bash
sudo cp deployment/nginx.conf /etc/nginx/sites-available/lam-api
sudo ln -sf /etc/nginx/sites-available/lam-api /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

## API Usage

### Starting the Service

**Development mode**:

```bash
cd /opt/LAM
source venv/bin/activate
python app_fastapi.py
```

**Production mode**:

```bash
sudo systemctl start lam-api
sudo systemctl status lam-api
```

### API Endpoints

#### Health Check

```bash
curl http://localhost:8000/health
```

#### List Available Motions

```bash
curl http://localhost:8000/motions
```

#### Process Image

```bash
curl -X POST "http://localhost:8000/process" \
  -F "image=@/path/to/image.jpg" \
  -F "motion_video=nice" \
  -F "enable_oac_file=true"
```

#### API Documentation

Access interactive API documentation at: `http://your-server-ip/docs`

### Using the Python Client

```bash
# Install client dependencies
pip install requests

# Process an image
python client_example.py \
  --image /path/to/image.jpg \
  --motion nice \
  --output-dir ./results \
  --base-url http://your-server-ip
```

## Monitoring and Management

### Service Management

```bash
# Start/stop/restart service
sudo systemctl start lam-api
sudo systemctl stop lam-api
sudo systemctl restart lam-api

# View service status
sudo systemctl status lam-api

# View logs
sudo journalctl -u lam-api -f
```

### Performance Monitoring

```bash
# GPU usage
nvidia-smi

# System resources
htop

# Disk usage
df -h
```

### Log Files

- **Service logs**: `sudo journalctl -u lam-api`
- **Nginx access logs**: `/var/log/nginx/lam-api.access.log`
- **Nginx error logs**: `/var/log/nginx/lam-api.error.log`

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

- Reduce batch size in config
- Use smaller render resolution
- Ensure no other GPU processes are running

**2. Model Loading Fails**

- Check model file paths and permissions
- Verify model files are downloaded correctly
- Check available disk space

**3. Service Won't Start**

- Check systemd service logs: `sudo journalctl -u lam-api`
- Verify virtual environment paths in service file
- Check CUDA installation: `nvidia-smi`

**4. Long Processing Times**

- Verify GPU is being used: `nvidia-smi`
- Check system resources: `htop`
- Ensure adequate cooling and power

### Performance Optimization

**1. GPU Memory**

```bash
# Add to environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

**2. CPU Parallelism**

```bash
export OMP_NUM_THREADS=8
export NUMBA_THREADING_LAYER=omp
```

**3. Model Caching**

- Keep models in memory between requests
- Use model quantization if available
- Implement model warming on startup

## Security Considerations

### Production Deployment

1. **Firewall Configuration**:

   ```bash
   sudo ufw allow 22    # SSH
   sudo ufw allow 80    # HTTP
   sudo ufw allow 443   # HTTPS
   sudo ufw enable
   ```

2. **SSL/TLS Setup**:

   ```bash
   # Install certbot
   sudo apt install certbot python3-certbot-nginx

   # Get SSL certificate
   sudo certbot --nginx -d your-domain.com
   ```

3. **Rate Limiting**:
   Add to nginx config:

   ```nginx
   limit_req_zone $binary_remote_addr zone=api:10m rate=10r/m;
   limit_req zone=api burst=5 nodelay;
   ```

4. **Authentication**:
   Consider adding API key authentication for production use.

## Scaling and Load Balancing

### Multiple GPU Setup

- Modify service to use multiple CUDA devices
- Load balance between GPU workers
- Use process pools for parallel inference

### Container Orchestration

- Use Kubernetes for large-scale deployment
- Implement horizontal pod autoscaling
- Use persistent volumes for model storage

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review service logs
3. Consult the original LAM documentation
4. Open an issue in the repository

## License

This deployment setup follows the same license as the LAM project.
