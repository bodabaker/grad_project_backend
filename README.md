# 🧠 AIoT Vision + MQTT + n8n Automation Stack

This project integrates multiple **Python-based computer vision microservices**, a **Mosquitto MQTT broker**, a **discovery beacon**, and an **n8n automation platform** — all running inside Docker. The stack enables automated workflows triggered by camera events and MQTT communication with other devices (e.g., ESP32, IoT lights).

---

## 🚀 Overview

| Component        | Purpose                                                                 |
|------------------|-------------------------------------------------------------------------|
| face-service     | Face recognition service using OpenCV + `face_recognition`.              |
| fire-service     | Fire detection service using YOLOv8.                                     |
| object-service   | Object detection service using OpenCV DNN (SSD MobileNet COCO).          |
| ocr-service      | OCR service using EasyOCR with tunable detection parameters.             |
| body-service     | Body/pose/fall detection service using YOLOv8 pose model.                |
| gesture-service  | Hand-based floor-zone gesture classifier (top=first, bottom=second).     |
| gemma-service    | Gemma 4 E2B local chat server powered by llama.cpp.                      |
| speaker-player   | PulseAudio/PipeWire playback API for uploaded or stored audio.           |
| mosquitto        | Lightweight MQTT broker for service communication.                       |
| server-beacon    | UDP broadcaster announcing the broker's IP for device auto-discovery.    |
| n8n              | Visual automation platform orchestrating workflows via MQTT/API triggers. |
| mediamtx         | RTSP/RTMP/HLS/WebRTC server for camera stream distribution.             |
| camera-publisher | FFmpeg container that captures webcam feed and publishes to RTSP.       |

Everything is self-contained and reproducible — no manual setup required.

---

## 🧩 Folder Structure

```
project-root/
├── app.py                    # Face detection service
├── beacon.py                 # UDP beacon for broker discovery
├── camera_publish.sh        # FFmpeg camera publishing script
├── docker-compose.yml       # Container orchestration
├── Dockerfile              # Face detection service build
├── mediamtx.yml           # MediaMTX server configuration
├── mosquitto.conf         # MQTT broker configuration
├── requirements.txt       # Python dependencies
├── captures/             # face snapshots
├── n8n_data/            # n8n database and configuration
│   ├── config/
│   ├── binaryData/
│   ├── git/
│   ├── nodes/
│   │   └── package.json
│   └── ssh/
├── persons/             # known people images
├── scripts/            # utility scripts
│   └── n8n-prune.sh    # database cleanup helper
├── speaker-player/     # PulseAudio/PipeWire audio playback service
│   ├── audio/          # stored playable audio files
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── .gitattributes
└── .gitignore
```

---

## ⚙️ Prerequisites & Setup

- **Docker** ≥ 24
- **Docker Compose** ≥ 2
- **Git** and **Git LFS**

Initialize Git LFS once:
```bash
git lfs install
git lfs track "n8n_data/database.sqlite"
git add .gitattributes
git commit -m "Track n8n DB via LFS"
```

Prepare directories and known faces:
```bash
mkdir -p persons captures n8n_data scripts
# Add images to persons/
persons/
├── alice.jpg
├── bob.jpg
```

### Faster-Whisper model (not committed to Git)
The ASR container mounts a local CTranslate2 Whisper checkpoint from `./models/whisper-medium` (see `asr-fastwhisper` service in `docker-compose.yml`). The model is ~1.4GB, so it's gitignored. Download it once on each machine:
```bash
mkdir -p models
git lfs install                         # ensures large files pull correctly
git clone https://huggingface.co/guillaumekln/whisper-medium-ct2 models/whisper-medium
# optional: set WHISPER_DEVICE=cuda in docker-compose.yml if you have an NVIDIA GPU
```

Build and start all containers:
```bash
docker compose up -d --build
```

### LLM (Ollama) model
The stack uses the instruction-tuned Qwen 2.5 7B model. Pull it inside the running `ollama` container (it persists in the `aiot_ollama_data` volume):
```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5:7b-instruct
docker compose exec ollama ollama list  # should show qwen2.5:7b-instruct
```
Quick sanity test:
```bash
curl -s http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5:7b-instruct","prompt":"1+1=","stream":false}'
```

Internal docker services:
- n8n workflows: [http://localhost:5678](http://localhost:5678)
- Face detection API: port 8000
- Fire detection API: port 8010
- Object detection API: port 8020
- OCR API: port 8030
- Body detection API: port 8040
- Gesture control API: port 8050
- Gemma 4 E2B API: port 8081
- Speaker playback API: port 5007
- MQTT broker: port 1883
- MediaMTX streaming:
  - RTSP: port 8554
  - RTMP: port 1935
  - HLS: port 8888
  - WebRTC: port 8889
- Piper TTS (Arabic, Kareem medium): port 5000
- Ollama (LLM host): port 11434
---


## 🔌 [PUBLIC] n8n Workflows & Webhooks

All public-facing API endpoints are now implemented as n8n workflows and triggered via HTTP webhooks.

Base URL: `http://<HOST IP>:5678/run`

**Available Webhooks:**
- `POST /camera-feed` — Get camera feed from RTSP stream
- `POST /door` — Trigger door open via MQTT broadcast
- `POST /agent` — Interact with AI agent (params: `sessionId`, `message`)
- `POST /voice` — Send audio file and receive AI agent audio response

### Scenario Automation Endpoints

The smart-home scenario CRUD/toggle API is exposed through n8n webhooks:

- `POST /scenarios/create` — Create and activate a scenario
- `GET /scenarios/get` — List all scenarios
- `POST /scenarios/update?id=<workflow-id>` — Update an existing scenario
- `DELETE /scenarios/delete?id=<workflow-id>` — Delete a scenario
- `POST /scenarios/toggle?id=<workflow-id>` — Activate/deactivate a scenario

### Input JSON Structure (Create / Update)

`POST /scenarios/create` and `POST /scenarios/update` accept the same JSON body:

```json
{
  "name": "My Automation",
  "trigger": {
    "type": "sensor",
    "sensor": "gas",
    "condition": "greater_than",
    "value": 400
  },
  "actions": [
    { "device": "buzzer", "action": "on" },
    { "device": "lights_rgb", "action": "c #FF0000" },
    { "delay": 10, "unit": "seconds" }
  ]
}
```

**Required top-level fields:**
- `name` (`string`): Human-readable scenario name
- `trigger` (`object`): Trigger definition (sensor or schedule)
- `actions` (`array`): Ordered list of device actions and/or delays

**Toggle body (`POST /scenarios/toggle`):**

```json
{ "active": true }
```

`active` is a required `boolean` (`true` to enable, `false` to disable).

**n8n Workflow Triggers:**
All endpoints trigger corresponding n8n workflows that orchestrate the logic, handle MQTT communication, and manage service calls.
---

## n8n Workflows

### Face Detection Workflow
This workflow handles face detection and recognition events:

**Triggers:**
- Manual execution
- MQTT message on topic `face/trigger/cmd`

**Process:**
1. Calls face detection service (`POST /detect-webcam`)
2. Analyzes detection results
3. Broadcasts recognition status via MQTT:
   - If face recognized: Sends person's name to `home/app/face-recognized`
   - If unknown face: Sends empty message to `home/app/face-unrecognized`

**Parameters:**
- Capture duration: 8 seconds
- Stops on first detection: Yes
- Uses `/data/persons` for known faces
- Saves captures to `/data/caps`

---
## 🤖 [PRIVATE] Face Detection API

Key endpoints:
- `GET /healthz` — Service health check
- `POST /detect-webcam` — Detect faces from webcam, save annotated frames
- `GET /stream` — MJPEG live stream

---

## 👁️ [PRIVATE] CV Services API Summary

### Fire Detection (`fire-service`, port 8010)
- `GET /healthz`
- `POST /detect-image`
- `POST /detect-webcam`
- `GET /stream`
- `GET /ui`

### Object Detection (`object-service`, port 8020)
- `GET /healthz`
- `POST /detect-image`
- `POST /detect-webcam`
- `GET /stream`
- `GET /ui`

### OCR (`ocr-service`, port 8030)
- `GET /healthz`
- `POST /detect-image`
- `POST /detect-webcam`
- `GET /stream`

### Body Detection (`body-service`, port 8040)
- `GET /healthz`
- `POST /detect-image`
- `POST /detect-webcam`
- `GET /stream`
- `GET /ui`

### Gesture Control (`gesture-service`, port 8050)
- `GET /healthz`
- `POST /detect-webcam`

### Speaker Playback (`speaker-player`, port 5007)
- `GET /healthz`
- `GET /files`
- `POST /play` — stream an uploaded audio file directly to the laptop speakers
- `POST /play-stored` — play a file stored inside the mounted audio folder

The playback service uses the host PulseAudio/PipeWire socket and blocks the request until playback completes.
If your host uses a non-default session path, set `XDG_RUNTIME_DIR`, `HOME`, `UID`, and `GID` before starting the stack so Docker can mount the correct socket and cookie.

Gesture response includes:
- `last_detected_floor` (`first_floor` or `second_floor`)
- `floor_hits`
- `detections_count`
- optional `timeline` (if `include_timeline=true`)

Example webcam detection:
```bash
curl -X POST http://localhost:8000/detect-webcam \
-F persons_dir=/data/persons \
-F stop_on_first=true \
-F max_seconds=10 \
-F annotated_dir=/data/caps \
-F frame_stride=1 \
-F tolerance=0.6 \
-F model=hog
```

## 📸 [PRIVATE] Camera Stream API

Key endpoints (base: `rtsp://localhost:8554`):
- `/cam` — Main camera RTSP stream endpoint

Access methods:
- RTSP direct: `rtsp://localhost:8554/cam`
- RTMP: `rtmp://localhost:1935/cam`
- HLS: `http://localhost:8888/cam`
- WebRTC: Available through port 8889

---

## 💬 MQTT Broker & Beacon

- Mosquitto runs on port 1883, accessible for local and LAN MQTT clients.
- The beacon (`beacon.py`) broadcasts the broker’s IP and responds to WHO_IS queries for device auto-discovery.

---

## 🔊 Piper Arabic TTS (HTTP)

- Service: `piper` container (voice: `ar_JO-kareem-medium`, sample rate 22050 Hz).
- Port: `5000` (HTTP).
 - Model files are mounted from `voices/ar_JO-kareem-medium/` (contains `model.onnx` and `model.json`).
 - Image is built locally from `piper-image/Dockerfile` (expects `piper-image/piper_amd64.tar.gz` pre-downloaded from https://github.com/rhasspy/piper/releases/tag/v1.2.0; override via build-arg `PIPER_TARBALL` if you name it differently).
 - The container runs a small FastAPI wrapper that shells out to the Piper binary bundled in the tarball.

Test synthesis:
```bash
curl -X POST http://localhost:5000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "مرحبا، أنا كريم وسوف أقرأ هذه الجملة."}' \
  --output output/piper_ar.wav
# plays a WAV using the Kareem voice
```

---

## 🧠 Managing n8n Data with Git + Git LFS

All n8n workflows, credentials, users, and executions are stored in `n8n_data/database.sqlite` and versioned via Git LFS. This enables instant backup, sync, and reproducible automation environments.

### Backup & Sync
Just commit and push as usual. The prune script is automatically run by the pre-commit hook, so you do not need to run it manually.
```bash
git add n8n_data/database.sqlite
git commit -m "Backup latest n8n state"
git push
```

### Remove n8n images before commit
If you want to remove execution images/files before committing, run:

```bash
PURGE_ALL_EXECUTIONS=1 REMOVE_BINARY_DATA=1 UNTRACK_RUNTIME_FILES=1 ./scripts/n8n-prune.sh
```

This command does four things safely:
- deletes execution rows from the n8n SQLite database
- removes generated files from `n8n_data/binaryData/`
- clears WAL/SHM + event logs
- removes those runtime artifacts from the Git index (without deleting tracked source files)

### Restore on a New Machine
```bash
git clone <your-repo>
cd <your-repo>
git lfs install
git lfs pull
docker compose up -d --build
```
n8n loads with all workflows, credentials, and users intact.

---

## 🧹 Pre-commit Hook Setup for New Users

To ensure the prune script runs automatically before each commit, new users must set up the pre-commit hook after cloning:

```bash
# Make sure the hook script exists and is executable
chmod +x .githooks/pre-commit
# Set the hooks path for your local repo
git config core.hooksPath .githooks
```

This only needs to be done once per clone.

---

## 🧱 Maintenance Commands

| Task                  | Command                          |
|-----------------------|----------------------------------|
| Start all containers  | docker compose up -d             |
| Stop all containers   | docker compose down              |
| View logs             | docker compose logs -f           |
| Prune n8n DB          | ./scripts/n8n-prune.sh           |
| Rebuild images        | docker compose build --no-cache  |

---

## 🧩 Summary

You now have a complete system that:
- Detects faces via face-service
- Broadcasts broker presence with server-beacon
- Syncs data in n8n via Git + LFS
- Is 100% reproducible and portable

Clone, pull, and run — no setup required 🎯

---
