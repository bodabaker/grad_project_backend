FROM jrottenberg/ffmpeg:6.1-ubuntu

ENV DEBIAN_FRONTEND=noninteractive
ENV VENV_PATH=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY camera-publisher/requirements.txt /app/requirements.txt
RUN python3 -m venv "$VENV_PATH" \
    && "$VENV_PATH/bin/pip" install --no-cache-dir --upgrade pip \
    && "$VENV_PATH/bin/pip" install --no-cache-dir -r /app/requirements.txt

COPY camera_publish.sh /app/publish.sh
COPY camera-publisher/frame_api.py /app/frame_api.py
COPY camera-publisher/start.sh /app/start.sh

RUN chmod +x /app/publish.sh /app/start.sh

EXPOSE 8060

CMD ["/app/start.sh"]
