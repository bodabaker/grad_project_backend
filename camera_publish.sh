#!/bin/sh
set -eu

# Defaults if env vars not provided
CAM_DEVICE="${CAM_DEVICE:-/dev/video0}"
CAM_SIZE="${CAM_SIZE:-1280x720}"
CAM_FPS="${CAM_FPS:-30}"
ENCODER="${ENCODER:-libx264}"        # on Raspberry Pi/ARM: h264_v4l2m2m
BITRATE="${BITRATE:-2500k}"
RTSP_URL="${RTSP_URL:-rtsp://mediamtx:8554/cam?publish=true}"

echo "[camera-publisher] Using device=$CAM_DEVICE size=$CAM_SIZE fps=$CAM_FPS encoder=$ENCODER bitrate=$BITRATE url=$RTSP_URL"

# Optional: uncomment to print supported formats
# ffmpeg -hide_banner -f v4l2 -list_formats all -i "$CAM_DEVICE" || true

while true; do
  echo "[camera-publisher] starting ffmpeg..."
  ffmpeg -hide_banner -loglevel info -nostdin \
    -f v4l2 -framerate "$CAM_FPS" -video_size "$CAM_SIZE" \
    -i "$CAM_DEVICE" \
    -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 \
    -c:v "$ENCODER" -preset veryfast -tune zerolatency -b:v "$BITRATE" \
    -pix_fmt yuv420p \
    -c:a aac -ar 44100 -ac 2 -shortest \
    -f rtsp -rtsp_transport tcp \
    "$RTSP_URL"
  rc=$?
  echo "[camera-publisher] ffmpeg exited with code $rc — retrying in 3s..."
  sleep 3
done

