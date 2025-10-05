# Face Detection Service

A FastAPI-based microservice for real-time face detection and recognition using a webcam or uploaded images. It leverages `face_recognition` and OpenCV for face detection, and provides a simple web UI for live viewing.

## Features
- Detect faces in images or webcam streams
- Recognize known persons from a directory of images
- Annotate and save processed frames
- REST API endpoints for integration
- MJPEG live stream endpoint for browser viewing
- Dockerized for easy deployment

## Requirements
- Python 3.11+
- See `requirements.txt` for Python dependencies
- System dependencies for OpenCV and dlib (see Dockerfile)

## Installation


### 1. Clone the repository
Clone your repository as usual. No need to change directories; all commands can be run from the project root.



### 2. No manual dependency or service startup needed
All dependencies are installed automatically inside the containers. You do not need to install anything on your host system.
You do not need to start any service manually—just use Docker Compose as described below.




### 3. Run with Docker Compose
Simply run:
```bash
docker compose up --build -d
```
This will build and start all containers as defined in `docker-compose.yml`. No other commands or manual service startup are required.

Main services started:
- **face-service**: FastAPI face detection API (exposes port 8000)
- **mosquitto**: MQTT broker (exposes port 1883)
- **broker-beacon**: Python beacon script
- **n8n**: Workflow automation (exposes port 5678)

## Usage

### 1. Prepare known faces
Place images of known persons in the `persons/` directory. Each image should contain one face and be named as the person's name (e.g., `alice.jpg`).

### 2. Start the service
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3. API Endpoints

#### Health Check
- `GET /healthz`

#### Detect Faces in Uploaded Image
- `POST /detect-image`
  - Form fields:
    - `persons_dir`: Path to known faces directory
    - `file`: Image file upload
    - `model`: "hog" (CPU) or "cnn" (GPU, if available)
    - `tolerance`: Recognition threshold (default: 0.6)
    - `annotated_out`: Optional path to save annotated image

#### Detect Faces from Webcam
- `POST /detect-webcam`
  - Form fields:
    - `persons_dir`: Path to known faces directory
    - `webcam`: Webcam index (default: 0)
    - `model`, `tolerance`, `max_seconds`, `max_frames`, `frame_stride`, `stop_on_first`, `annotated_dir`, `save_all_frames`, `include_timeline`

#### MJPEG Live Stream
- `GET /stream`
  - Query params:
    - `persons_dir`, `webcam`, `model`, `tolerance`, `frame_stride`, `annotated`

#### Web UI
- `GET /ui`
  - Simple browser interface for live viewing and controls

## Example

1. Place images in `persons/`:
   - `persons/alice.jpg`
   - `persons/bob.jpg`
2. Start the service
3. Open [http://localhost:8000/ui](http://localhost:8000/ui) in your browser

## Project Structure
```
app.py                # Main FastAPI app
requirements.txt      # Python dependencies
Dockerfile            # Container setup
persons/              # Known faces
captures/             # Saved frames (optional)
```

## License
MIT
