FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg sox tini pkg-config build-essential \
      libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
      libavfilter-dev libswscale-dev libswresample-dev && \
    rm -rf /var/lib/apt/lists/*

# CPU by default. If you plan CUDA, switch to appropriate wheels in requirements.
# Install AV that matches FFmpeg 5/6 APIs, then install faster-whisper without pulling its av pin.
RUN pip install --no-cache-dir \
      fastapi \
      uvicorn[standard] \
      av==12.3.0 \
      pydub \
      python-multipart \
      numpy==1.26.4 \
      ctranslate2==4.6.1 \
      onnxruntime==1.23.2 \
      tokenizers==0.15.2 \
      huggingface-hub==0.36.0 \
      tqdm==4.67.1 && \
    pip install --no-cache-dir --no-deps faster-whisper==1.0.0

COPY asr_server.py /app/asr_server.py

ENV ASR_PORT=5003 \
    WHISPER_MODEL=small \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE=float32 \
    VAD_FILTER=true

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["uvicorn","asr_server:app","--host","0.0.0.0","--port","5003"]
