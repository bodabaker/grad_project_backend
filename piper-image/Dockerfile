FROM python:3.11-slim

# Minimal runtime deps for Piper binary
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl libespeak-ng1 \
 && rm -rf /var/lib/apt/lists/*

# Copy pre-downloaded Piper binary archive into the image.
# Download manually to build context as: piper-image/piper_amd64.tar.gz (or override via build-arg)
ARG PIPER_TARBALL=piper_amd64.tar.gz
COPY ${PIPER_TARBALL} /tmp/piper.tar.gz

# Extract Piper bundle to /opt/piper
RUN set -eux; \
  mkdir -p /opt/piper; \
  tar -xzf /tmp/piper.tar.gz -C /opt; \
  rm /tmp/piper.tar.gz; \
  chmod +x /opt/piper/piper

ENV LD_LIBRARY_PATH=/opt/piper:${LD_LIBRARY_PATH}
ENV PATH=/opt/piper:${PATH}

WORKDIR /app

# Install lightweight HTTP server deps
RUN pip install --no-cache-dir fastapi uvicorn[standard]

# Copy HTTP server
COPY server.py /app/server.py

ENTRYPOINT ["uvicorn", "server:app", "--host", "0.0.0.0"]
