FROM ghcr.io/mostlygeek/llama-swap:cuda

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY llama-swap/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir -r /tmp/requirements.txt

COPY llama-swap/lahgtna_wrapper.py /app/lahgtna_wrapper.py

ENV HF_HOME=/models/hf-cache
ENV TRANSFORMERS_CACHE=/models/hf-cache
