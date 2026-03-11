#!/bin/bash
# Fish-speech API server with reduced VRAM usage
# Optimized for RTX 3050 6GB (shared with llama.cpp)

cd /home/boda/Projects/graduation_project/fish-speech || exit 1

# Kill existing instances
pkill -f "api_server.py" || true
sleep 2

# Start with reduced VRAM settings:
# --compile: Disable torch.compile to save VRAM
# --max-text-length 200: Reduce max input length
# --workers 1: Single worker to minimize memory
exec uv run tools/api_server.py \
  --listen 0.0.0.0:8080 \
  --llama-checkpoint-path checkpoints/openaudio-s1-mini \
  --decoder-checkpoint-path checkpoints/openaudio-s1-mini/codec.pth \
  --decoder-config-name modded_dac_vq \
  --half \
  --workers 1 \
  --compile disable \
  --max-text-length 200
