#!/bin/sh
set -eu

# Start RTSP publisher loop in background
/app/publish.sh &
PUBLISH_PID=$!

cleanup() {
  kill "$PUBLISH_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

# Start frame API in foreground
exec uvicorn frame_api:app --host 0.0.0.0 --port "${FRAME_API_PORT:-8060}"
