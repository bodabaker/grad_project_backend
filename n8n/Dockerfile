FROM alpine:3.22 AS mqtt-tools
RUN apk add --no-cache mosquitto-clients

FROM python:3.11-alpine AS python-runtime

FROM n8nio/n8n:2.17.7

USER root

# Download static ffmpeg binary (musl-compatible)

# Download static ffmpeg binary (musl-compatible)
RUN wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -O /tmp/ffmpeg.tar.xz && \
    tar -xf /tmp/ffmpeg.tar.xz -C /tmp && \
    cp /tmp/ffmpeg-*/ffmpeg /usr/bin/ffmpeg && \
    chmod +x /usr/bin/ffmpeg && \
    rm -rf /tmp/ffmpeg*

COPY --from=python-runtime /usr/local/ /usr/local/

COPY homectl/ /homectl/

RUN pip install --no-cache-dir -e /homectl

COPY --from=mqtt-tools /usr/bin/mosquitto_sub /usr/local/bin/mosquitto_sub
COPY --from=mqtt-tools /usr/lib/libmosquitto.so.1 /usr/lib/libmosquitto.so.1
COPY --from=mqtt-tools /usr/lib/libcjson.so.1 /usr/lib/libcjson.so.1
COPY --from=mqtt-tools /usr/lib/libssl.so.3 /usr/lib/libssl.so.3
COPY --from=mqtt-tools /usr/lib/libcrypto.so.3 /usr/lib/libcrypto.so.3
COPY --from=mqtt-tools /usr/lib/libcares.so.2 /usr/lib/libcares.so.2

# Install custom AI Agent V2 node
COPY n8n_data/nodes/n8n-nodes-ai-agent-v2 /tmp/n8n-nodes-ai-agent-v2
RUN mkdir -p /home/node/.n8n/custom && \
    cp -r /tmp/n8n-nodes-ai-agent-v2 /home/node/.n8n/custom/ && \
    chown -R node:node /home/node/.n8n/custom

USER node
