# syntax=docker/dockerfile:1
# Multi-stage build for omnivoice.cpp TTS service
# -----------------------------------------------------------------------------
# Stage 1: Build omnivoice.cpp binaries with CUDA support
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --recurse-submodules https://github.com/ServeurpersoCom/omnivoice.cpp.git

WORKDIR /src/omnivoice.cpp
# Apply patches: --server mode, --steps, --offload-encode
COPY omnivoice-service/omnivoice-tts.patch /tmp/omnivoice-tts.patch
RUN git apply /tmp/omnivoice-tts.patch
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf && ldconfig
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}
RUN rm -rf build && mkdir build && cd build && \
    cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
             -DCMAKE_CUDA_ARCHITECTURES=86 \
             -DGGML_CUDA_FA_ALL_QUANTS=ON \
             -DGGML_CUDA_GRAPHS=ON && \
    cmake --build . --config Release -j "$(nproc)"

# -----------------------------------------------------------------------------
# Stage 2: Runtime with Python + FastAPI
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY omnivoice-service/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Copy built binaries and backends
COPY --from=builder /src/omnivoice.cpp/build/omnivoice-tts /usr/local/bin/omnivoice-tts
COPY --from=builder /src/omnivoice.cpp/build/omnivoice-codec /usr/local/bin/omnivoice-codec
COPY --from=builder /src/omnivoice.cpp/build/*.so /usr/local/lib/

ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/bin:${PATH}
ENV OMNIVOICE_CODEC_BACKEND=

WORKDIR /app
COPY omnivoice-service/app.py /app/app.py

EXPOSE 5008

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5008"]
