# 🧠 Face Detection + MQTT + n8n Automation Stack

This project integrates a **Python-based face detection microservice**, a **Mosquitto MQTT broker**, a **discovery beacon**, and an **n8n automation platform** — all running inside Docker. The stack enables automated workflows triggered by face recognition events and MQTT communication with other devices (e.g., ESP32, IoT lights).

---

## 🚀 Overview

| Component        | Purpose                                                                 |
|------------------|-------------------------------------------------------------------------|
| face-service     | FastAPI container using OpenCV & face_recognition for webcam detection.  |
| mosquitto        | Lightweight MQTT broker for service communication.                       |
| broker-beacon    | UDP broadcaster announcing the broker’s IP for device auto-discovery.    |
| n8n              | Visual automation platform orchestrating workflows via MQTT/API triggers. |

Everything is self-contained and reproducible — no manual setup required.

---

## 🧩 Folder Structure

```
project-root/
├── docker-compose.yml
├── face_service/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── n8n_data/
│   └── database.sqlite        # stored & versioned via Git LFS
├── captures/                  # face snapshots
├── persons/                   # known people
├── beacon.py                  # UDP beacon for broker discovery
├── scripts/
│   └── n8n-prune.sh           # database cleanup helper
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
├── alice/1.jpg
├── bob/1.jpg
```

Build and start all containers:
```bash
docker compose up -d --build
```

Access services:
- Face detection API: [http://localhost:8000](http://localhost:8000)
- n8n automation: [http://localhost:5678](http://localhost:5678)
- MQTT broker: port 1883

---

## 🤖 Face Detection API

Key endpoints:
- `GET /healthz` — Service health check
- `POST /detect-webcam` — Detect faces from webcam, save annotated frames
- `POST /detect-image` — Detect faces in uploaded image
- `GET /stream` — MJPEG live stream
- `GET /ui` — Simple browser interface

Example webcam detection:
```bash
curl -X POST http://localhost:8000/detect-webcam \
  -F persons_dir=/data/persons \
  -F webcam=0 \
  -F max_seconds=8 \
  -F annotated_dir=/data/caps
```

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
