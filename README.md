# 🧠 Face Detection + MQTT + n8n Automation Stack

This project integrates a **Python-based face detection microservice**, a **Mosquitto MQTT broker**, a **discovery beacon**, and an **n8n automation platform** — all running inside Docker. The stack enables automated workflows triggered by face recognition events and MQTT communication with other devices (e.g., ESP32, IoT lights).

---

## 🚀 Overview

| Component        | Purpose                                                                 |
|------------------|-------------------------------------------------------------------------|
| face-service     | FastAPI container using OpenCV & face_recognition for webcam detection.  |
| mosquitto        | Lightweight MQTT broker for service communication.                       |
| broker-beacon    | UDP broadcaster announcing the broker's IP for device auto-discovery.    |
| n8n              | Visual automation platform orchestrating workflows via MQTT/API triggers. |
| mediamtx         | RTSP/RTMP/HLS/WebRTC server for camera stream distribution.             |
| camera-publisher | FFmpeg container that captures webcam feed and publishes to RTSP.       |
| face-service     | Face recognition service that consumes RTSP stream for processing.      |

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

Build and start all containers:
```bash
docker compose up -d --build
```

Internal docker services:
- n8n workflows: [http://localhost:5678](http://localhost:5678)
- Face detection API: port 8000
- MQTT broker: port 1883
- MediaMTX streaming:
  - RTSP: port 8554
  - RTMP: port 1935
  - HLS: port 8888
  - WebRTC: port 8889
---


## 🔌 [PUBLIC] n8n API Endpoints

Key api URLs (base: `http://<HOST IP>:5678`):
- `/api/camera-feed` - [Get] — Redirects to docker container that streams camera feed
- `/api/door` - [POST] — Broadcasts MQTT message to open door
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

Example webcam detection:
```bash
curl -X POST http://localhost:8000/detect-webcam \
-F persons_dir=/data/persons \
-F stop_on_first=true \
-F max_seconds=10 \
-F annotated_dir=/data/caps \
-F frame_stride=3 \
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

## 🧠 Managing n8n Data with Git + Git LFS

All n8n workflows, credentials, users, and executions are stored in `n8n_data/database.sqlite` and versioned via Git LFS. This enables instant backup, sync, and reproducible automation environments.

### Backup & Sync
Just commit and push as usual. The prune script is automatically run by the pre-commit hook, so you do not need to run it manually.
```bash
git add n8n_data/database.sqlite
git commit -m "Backup latest n8n state"
git push
```

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
- Broadcasts broker presence with broker-beacon
- Syncs data in n8n via Git + LFS
- Is 100% reproducible and portable

Clone, pull, and run — no setup required 🎯

---
