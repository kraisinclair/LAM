# LAM FastAPI - Image to ZIP Conversion API

This FastAPI application provides a REST API interface for the LAM (Large Avatar Model) system, allowing you to convert images into animatable avatar ZIP files.

## Features

- **Image Upload**: Upload face images via HTTP POST
- **Motion Selection**: Choose from available motion sequences
- **ZIP Generation**: Automatically creates ZIP files compatible with Open Avatar Chat
- **Video Generation**: Creates animated videos with audio
- **File Downloads**: Download processed images, videos, and ZIP files

## API Endpoints

### 1. Health Check

```
GET /
```

Returns API status and availability.

### 2. List Available Motions

```
GET /motions
```

Returns a list of available motion sequences.

### 3. Process Image

```
POST /process
```

Upload an image and create avatar assets.

**Parameters:**

- `image` (file): Input image file (PNG, JPG, JPEG)
- `motion` (string): Motion sequence name (default: "nice")

**Response:**

```json
{
  "status": "success",
  "message": "Image processed successfully",
  "results": {
    "processed_image": "/path/to/processed_image.png",
    "video": "/path/to/output_video.mp4",
    "zip_file": "/path/to/avatar.zip"
  }
}
```

### 4. Download Files

```
GET /download/{file_type}/{filename}
```

Download generated files.

**Parameters:**

- `file_type`: Type of file (image, video, zip)
- `filename`: Name of the file to download

## Setup and Installation

### Prerequisites

1. **Hardware Requirements:**

   - NVIDIA GPU with CUDA support
   - Minimum 8GB GPU memory

2. **Software Requirements:**
   - Python 3.8+
   - CUDA 11.8 or 12.1
   - Blender (for GLB generation)

### Installation Steps

1. **Install Dependencies:**

   ```bash
   pip install -r requirements_fastapi.txt
   ```

2. **Download Model Files:**
   Make sure you have the LAM model files in the correct location:

   ```
   ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
   ./model_zoo/flame_tracking_models/
   ```

3. **Prepare Assets:**
   Ensure motion sequences are available:
   ```
   ./assets/sample_motion/export/
   ./assets/sample_oac/
   ```

### Running the Server

1. **Start the FastAPI Server:**

   ```bash
   python app_fastapi.py
   ```

2. **Alternative with Uvicorn:**

   ```bash
   uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Access the API:**
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/

## Usage Examples

### Python Example

```python
import requests

# Upload and process an image
with open('my_image.jpg', 'rb') as f:
    files = {'image': f}
    data = {'motion': 'nice'}
    response = requests.post('http://localhost:8000/process', files=files, data=data)

result = response.json()
print(f"ZIP file: {result['results']['zip_file']}")

# Download the ZIP file
zip_filename = result['results']['zip_file'].split('/')[-1]
zip_response = requests.get(f'http://localhost:8000/download/zip/{zip_filename}')
with open('downloaded_avatar.zip', 'wb') as f:
    f.write(zip_response.content)
```

### cURL Examples

1. **Check API Status:**

   ```bash
   curl http://localhost:8000/
   ```

2. **List Available Motions:**

   ```bash
   curl http://localhost:8000/motions
   ```

3. **Process an Image:**

   ```bash
   curl -X POST http://localhost:8000/process \
     -F 'image=@assets/sample_input/james.png' \
     -F 'motion=nice'
   ```

4. **Download a ZIP File:**
   ```bash
   curl -O http://localhost:8000/download/zip/uploaded_image_nice.zip
   ```

### Test Script

Use the provided test script:

```bash
# Run basic test
python example_api_usage.py

# Test with custom image and motion
python example_api_usage.py --image my_image.jpg --motion happy

# Show only cURL examples
python example_api_usage.py --curl-only
```

## Configuration

### Environment Variables

Set these environment variables to customize the API:

```bash
export APP_MODEL_NAME="./model_zoo/lam_models/releases/lam/lam-20k/step_045500/"
export APP_INFER="./configs/inference/lam-20k-8gpu.yaml"
export BLENDER_PATH="/path/to/blender"
```

### Output Directory

Generated files are saved to:

```
./output/api_exports/
├── uploaded_image_nice.zip
├── uploaded_image_nice.mp4
└── uploaded_image_nice_processed.png
```

## API Documentation

Once the server is running, visit http://localhost:8000/docs for interactive API documentation powered by Swagger UI.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**

   - Reduce batch size or use a GPU with more memory
   - Make sure no other processes are using the GPU

2. **Model Files Not Found:**

   - Check that model files are downloaded and in the correct paths
   - Verify the `APP_MODEL_NAME` environment variable

3. **Blender Not Found:**

   - Install Blender and set the `BLENDER_PATH` correctly
   - Make sure Blender is executable

4. **Motion Not Found:**
   - Check available motions using `GET /motions`
   - Ensure motion files exist in `./assets/sample_motion/export/`

### Performance Tips

- **First Request Latency:** The first request may take longer due to model loading
- **Concurrent Requests:** The API processes requests sequentially due to GPU memory constraints
- **File Cleanup:** Old files in `./output/api_exports/` should be cleaned up periodically

## Integration

### Open Avatar Chat

The generated ZIP files are compatible with Open Avatar Chat applications. They contain:

- `skin.glb`: 3D avatar mesh
- `animation.glb`: Animation data
- `offset.ply`: Point cloud data

### Custom Applications

You can integrate this API into your own applications by making HTTP requests to the endpoints. The API follows REST conventions and returns JSON responses.

## License

This project follows the same license as the original LAM project (Apache License 2.0).
