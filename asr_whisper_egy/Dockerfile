FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py

ENV ASR_PORT=5004 \
    HF_WHISPER_MODEL=MAdel121/whisper-small-egyptian-arabic \
    ASR_LANGUAGE=ar \
    ASR_CHUNK_LENGTH_S=20 \
    ASR_BATCH_SIZE=8 \
    HF_HOME=/models/hf-cache \
    CUDA_VISIBLE_DEVICES=""

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5004"]
