# ==============================================================================
# High-Performance CUDA 12.8 Dockerfile for PrismML Bonsai 2 27B
# Uses official pre-compiled PrismML Linux CUDA 12.8 binaries for instant builds
# Supports: RTX 5090 (Blackwell), RTX 6000 Ada / L40S, A40 / A6000 (Ampere)
# ==============================================================================

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python-is-python3 \
    curl \
    ca-certificates \
    tar \
    gzip \
    libgomp1 \
    libcurl4 \
    procps \
    pciutils \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Change this value when RunPod needs to be forced onto a fresh source/build
# revision instead of reusing a previously cached repository or Docker layer.
ARG BUILD_CACHE_BUSTER="v1.0.3"
RUN echo "[Build] Cache buster: ${BUILD_CACHE_BUSTER}" > /app/.build-cache-buster

# Download and install pre-compiled PrismML llama.cpp with CUDA 12.8 and Ternary Bonsai 2 support
ARG LLAMA_PRISM_RELEASE="https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b10787-73bfe6c/llama-prism-b10787-73bfe6c-bin-linux-cuda-12.8-x64.tar.gz"

RUN mkdir -p /opt/llama.cpp && \
    echo "[Build] Downloading pre-compiled PrismML CUDA 12.8 binaries..." && \
    curl -sL "$LLAMA_PRISM_RELEASE" | tar -xz -C /opt/llama.cpp && \
    find /opt/llama.cpp -name "llama-server" -exec cp {} /usr/local/bin/llama-server \; && \
    find /opt/llama.cpp -name "llama-cli" -exec cp {} /usr/local/bin/llama-cli \; && \
    chmod +x /usr/local/bin/llama-server /usr/local/bin/llama-cli && \
    echo "/opt/llama.cpp" >> /etc/ld.so.conf.d/llama.conf && \
    find /opt/llama.cpp -type d -name "lib*" -exec echo {} >> /etc/ld.so.conf.d/llama.conf \; && \
    find /opt/llama.cpp -type d -name "bin*" -exec echo {} >> /etc/ld.so.conf.d/llama.conf \; && \
    ldconfig

ENV LD_LIBRARY_PATH="/opt/llama.cpp:/opt/llama.cpp/bin:/opt/llama.cpp/lib:${LD_LIBRARY_PATH}"

# Install RunPod Serverless SDK and runtime utilities
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy entrypoint scripts and handlers across standard RunPod paths
COPY entrypoint.sh /app/entrypoint.sh
COPY handler.py /app/handler.py
COPY handler.py /handler.py
COPY rp_handler.py /app/rp_handler.py
COPY rp_handler.py /rp_handler.py
COPY test_input.json /app/test_input.json
COPY test_input.json /test_input.json
COPY src /app/src
COPY .runpod /app/.runpod
RUN chmod +x /app/entrypoint.sh /app/handler.py /handler.py /app/rp_handler.py /rp_handler.py /app/src/handler.py /app/src/rp_handler.py

# Create model directory
RUN mkdir -p /workspace/models

# Default Runtime Configuration
ENV MODEL_PATH="/workspace/models/Ternary-Bonsai-2-27B-PQ2_0.gguf" \
    MMPROJ_PATH="/workspace/models/Ternary-Bonsai-2-27B-mmproj-BF16.gguf" \
    HF_REPO_ID="prism-ml/Ternary-Bonsai-2-27B-gguf" \
    HF_MODEL_FILE="Ternary-Bonsai-2-27B-PQ2_0.gguf" \
    HF_MMPROJ_FILE="Ternary-Bonsai-2-27B-mmproj-BF16.gguf" \
    HOST="0.0.0.0" \
    PORT=8000 \
    CTX_SIZE="261120" \
    PARALLEL_SLOTS="6" \
    KV_CACHE_TYPE_K="q8_0" \
    KV_CACHE_TYPE_V="q8_0" \
    BATCH_SIZE=4096 \
    UBATCH_SIZE=1024 \
    FLASH_ATTENTION=true \
    MAX_TOKENS=8192 \
    N_GPU_LAYERS=99 \
    RUNPOD_MODE="auto"

EXPOSE 8000

CMD ["python3", "-u", "/handler.py"]
