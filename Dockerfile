# ==============================================================================
# Multi-Architecture CUDA 12.8 Dockerfile for PrismML Bonsai 2 27B
# Supports:
#   - NVIDIA RTX 5090 (Blackwell sm_100 / sm_120)
#   - NVIDIA RTX 6000 Ada / L40S / RTX 4090 (Ada Lovelace sm_89)
#   - NVIDIA A40 / A100 / RTX A6000 (Ampere sm_80 / sm_86)
# Features:
#   - 256k Context Window support with quantized KV Cache (q8_0 / q4_0)
#   - Native Ternary 2-bit (PQ2_0) and multi-quant Flash Attention
#   - Multimodal Vision Projector (mmproj) support
#   - RunPod Serverless and Pod compatible with OpenAI API (/v1/chat/completions)
# ==============================================================================

# Stage 1: Build llama.cpp with CUDA 12.8 & Multi-Architecture Target
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_CUDA_ARCHITECTURES="80;86;89;90;100;120"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ninja-build \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Configurable build worker concurrency (defaults to all CPU cores)
ARG MAX_JOBS=""
ARG NVCC_THREADS="0"

WORKDIR /build

# Fast shallow clone with submodules
RUN git clone --depth 1 --shallow-submodules --recurse-submodules https://github.com/ggml-org/llama.cpp.git .

# Build with maximum parallel workers:
# - Ninja multi-core job execution (-j)
# - NVCC multi-target parallel compilation (--threads 0) across all SM architectures (80..120)
RUN JOBS="${MAX_JOBS:-$(nproc)}" && \
    echo "[Build] Compiling llama.cpp with $JOBS parallel CPU workers and NVCC multi-arch threads..." && \
    cmake -B build -G Ninja \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;100;120" \
        -DCMAKE_CUDA_FLAGS="--threads ${NVCC_THREADS} -O3" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_BUILD_PARALLEL_LEVEL="$JOBS" \
        -DLLAMA_CURL=ON && \
    cmake --build build --config Release --parallel "$JOBS"

# Stage 2: Minimal High-Performance Runtime
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    ca-certificates \
    libgomp1 \
    libcurl4 \
    procps \
    pciutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy compiled llama binaries from builder
COPY --from=builder /build/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /build/build/bin/llama-cli /usr/local/bin/llama-cli

# Install RunPod Serverless SDK and lightweight proxy utilities
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy launcher scripts and handlers
COPY entrypoint.sh /app/entrypoint.sh
COPY handler.py /app/handler.py
COPY rp_handler.py /app/rp_handler.py
COPY test_input.json /app/test_input.json
COPY src /app/src
COPY .runpod /app/.runpod
RUN chmod +x /app/entrypoint.sh /app/handler.py /app/rp_handler.py /app/src/handler.py /app/src/rp_handler.py

# Create model storage directory
RUN mkdir -p /workspace/models

# Default Runtime Configuration
ENV MODEL_PATH="/workspace/models/Ternary-Bonsai-2-27B-PQ2_0.gguf" \
    MMPROJ_PATH="/workspace/models/mmproj-bonsai-2-f16.gguf" \
    HF_REPO_ID="" \
    HF_MODEL_FILE="" \
    HF_MMPROJ_FILE="" \
    HOST="0.0.0.0" \
    PORT=8000 \
    CTX_SIZE="auto" \
    PARALLEL_SLOTS="auto" \
    KV_CACHE_TYPE_K="auto" \
    KV_CACHE_TYPE_V="auto" \
    BATCH_SIZE=4096 \
    UBATCH_SIZE=1024 \
    FLASH_ATTENTION=true \
    MAX_TOKENS=8192 \
    N_GPU_LAYERS=99 \
    RUNPOD_MODE="auto"

EXPOSE 8000

CMD ["python3", "-u", "/app/handler.py"]
ENTRYPOINT ["/app/entrypoint.sh"]