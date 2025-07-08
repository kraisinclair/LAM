# LAM FastAPI Quick Start

Get the LAM API running on Ubuntu server in minutes!

## 🚀 Quick Deployment

### Prerequisites

- Ubuntu 20.04/22.04 server with NVIDIA GPU
- Sudo access
- At least 16GB RAM and 50GB storage

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd LAM
chmod +x deployment/deploy_ubuntu.sh
./deployment/deploy_ubuntu.sh
```

### 2. Download Models

Download the required model files to these directories:

```
model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors
model_zoo/flame_tracking_models/68_keypoints_model.pkl
model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd
model_zoo/flame_tracking_models/matting/stylematte_synth.pt
model_zoo/flame_tracking_models/FaceBoxesV2.pth
assets/sample_motion/export/nice/flame_param/
assets/sample_motion/export/nice/nice.wav
assets/sample_oac/template_file.fbx
assets/sample_oac/animation.glb
```

### 3. Start Service

```bash
sudo systemctl start lam-api
sudo systemctl status lam-api
```

### 4. Test API

```bash
# Check if API is running
curl http://localhost:8000/health

# List available motions
curl http://localhost:8000/motions

# Process an image
python client_example.py --image path/to/your/image.jpg
```

## 📡 API Endpoints

| Endpoint                        | Method | Description                      |
| ------------------------------- | ------ | -------------------------------- |
| `/`                             | GET    | API status and available motions |
| `/health`                       | GET    | Health check                     |
| `/motions`                      | GET    | List available motion videos     |
| `/process`                      | POST   | Process image (upload + motion)  |
| `/download/{job_id}/{filename}` | GET    | Download processed files         |
| `/docs`                         | GET    | Interactive API documentation    |

## 🔧 Docker Alternative

For containerized deployment:

### Quick Docker Setup

```bash
# Install Docker with GPU support
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Build the image (with automatic fallback)
chmod +x build_docker.sh
./build_docker.sh

# Run with docker-compose
docker-compose up -d
```

### Build Options

If you encounter CUDA compilation errors:

```bash
# Try the smart build script (recommended)
./build_docker.sh

# Or build minimal version (faster, fewer features)
./build_docker.sh --minimal

# Or build and test
./build_docker.sh --test
```

### Troubleshooting Docker Build

**CUDA compilation errors?**

```bash
# Use minimal build (skips problematic CUDA packages)
./build_docker.sh --minimal
```

**Build too slow?**

```bash
# Direct build with original Dockerfile
docker build -t lam-api:latest .
```

## 🧪 Testing

### Python Client

```bash
pip install requests

python client_example.py \
  --image /path/to/image.jpg \
  --motion nice \
  --output-dir ./results
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/process" \
  -F "image=@/path/to/image.jpg" \
  -F "motion_video=nice" \
  -F "enable_oac_file=true"
```

### JavaScript/Browser

```javascript
const formData = new FormData();
formData.append("image", imageFile);
formData.append("motion_video", "nice");
formData.append("enable_oac_file", "true");

fetch("http://your-server:8000/process", {
  method: "POST",
  body: formData,
})
  .then((response) => response.json())
  .then((data) => console.log(data));
```

## 🔍 Monitoring

```bash
# Service status
sudo systemctl status lam-api

# View logs
sudo journalctl -u lam-api -f

# GPU usage
nvidia-smi

# System resources
htop
```

## 🔧 Configuration

Edit `/opt/LAM/deployment/lam-api.service` to:

- Change GPU device: `CUDA_VISIBLE_DEVICES=0`
- Adjust memory settings
- Modify user/group permissions

## 🆘 Troubleshooting

**Service won't start?**

```bash
sudo journalctl -u lam-api
# Check CUDA: nvidia-smi
# Check disk space: df -h
```

**Out of memory?**

- Reduce image resolution
- Check other GPU processes: `nvidia-smi`
- Restart service: `sudo systemctl restart lam-api`

**Slow processing?**

- Verify GPU usage: `nvidia-smi`
- Check CPU usage: `htop`
- Ensure adequate cooling

## 📚 Full Documentation

For complete setup instructions, see [README_API_DEPLOYMENT.md](README_API_DEPLOYMENT.md)
