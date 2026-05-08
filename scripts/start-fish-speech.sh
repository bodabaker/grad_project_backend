#!/bin/bash
# Fish Speech API server with minimal VRAM usage on CUDA
# Optimized for RTX 3050 6GB while keeping llama on GPU and decoder on CPU

cd /home/boda/Projects/graduation_project/aiot/fish-speech || exit 1

# Kill existing instances
pkill -f "api_server.py" || true
sleep 2

# Start with CUDA mode and reduced VRAM settings:
# --max-text-length 50: Reduce max input length  
# --workers 1: Single worker to minimize memory
# --device cuda: Keep the semantic model on the GPU
exec /app/.venv/bin/python /app/fish-speech-api-wrapper.py \
  --listen 0.0.0.0:8080 \
  --llama-checkpoint-path checkpoints/fish-speech-1.5 \
  --device cuda \
  --half \
  --workers 1 \
  --max-text-length 50
