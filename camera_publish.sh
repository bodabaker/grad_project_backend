#!/bin/sh
set -eu

# Defaults if env vars not provided
CAM_DEVICE="${CAM_DEVICE:-/dev/video0}"
CAM_SIZE="${CAM_SIZE:-1280x720}"
CAM_FPS="${CAM_FPS:-30}"
ENCODER="${ENCODER:-libx264}"        # on Raspberry Pi/ARM: h264_v4l2m2m
BITRATE="${BITRATE:-2500k}"
RTSP_URL="${RTSP_URL:-rtsp://mediamtx:8554/cam}"

echo "[camera-publisher] Using device=$CAM_DEVICE size=$CAM_SIZE fps=$CAM_FPS encoder=$ENCODER bitrate=$BITRATE url=$RTSP_URL"

# Optional: uncomment to print supported formats
# ffmpeg -hide_banner -f v4l2 -list_formats all -i "$CAM_DEVICE" || true

while true; do
  echo "[camera-publisher] starting ffmpeg..."
  ffmpeg -hide_banner -loglevel info -nostdin \
    -f v4l2 -input_format yuyv422 -framerate "$CAM_FPS" -video_size "$CAM_SIZE" \
    -i "$CAM_DEVICE" \
    -c:v "$ENCODER" -preset ultrafast -tune zerolatency -b:v "$BITRATE" \
    -pix_fmt yuv420p -profile:v baseline \
    -g "$CAM_FPS" -keyint_min "$CAM_FPS" \
    -max_delay 0 -bf 0 -bufsize "${BITRATE%k}k" \
    -x264opts "no-scenecut:nal-hrd=cbr" \
    -flags low_delay -probesize 32 -analyzeduration 0 \
    -fflags nobuffer -flags low_delay \
    -an \
    -f rtsp -rtsp_transport tcp \
    "$RTSP_URL"
  rc=$?
  echo "[camera-publisher] ffmpeg exited with code $rc — retrying in 3s..."
  sleep 3
done

