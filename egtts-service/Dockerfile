FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    sox \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY egtts-service/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

COPY egtts-service/app.py /app/app.py

EXPOSE 5004

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5004"]
