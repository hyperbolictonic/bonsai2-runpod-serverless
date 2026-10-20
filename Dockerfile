# Build the same PrismML CUDA runtime used by the tested Hexadi worker, pinned
# to the release commit so a later upstream branch move cannot silently change it.
ARG CUDA_VERSION=12.8.1
ARG LLAMA_CPP_COMMIT=73bfe6cc485554822f80c473a3c0c4f25e80f41b

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder
ARG LLAMA_CPP_COMMIT
ARG CUDA_ARCHITECTURES="80;120"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone https://github.com/PrismML-Eng/llama.cpp.git \
    && cd llama.cpp \
    && git checkout "${LLAMA_CPP_COMMIT}"

WORKDIR /src/llama.cpp
# Docker has no NVIDIA driver at build time; CUDA's stub is link-only.
RUN test -f /usr/local/cuda/lib64/stubs/libcuda.so \
    && ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs

RUN cmake -S . -B build \
      -DGGML_CUDA=ON \
      -DGGML_NATIVE=OFF \
      -DLLAMA_CURL=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
      -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
      -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
    && cmake --build build --config Release -j"$(nproc)" --target llama-server

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_REPO=prism-ml/Ternary-Bonsai-2-27B-gguf \
    MODEL_FILE=Ternary-Bonsai-2-27B-PQ2_0.gguf \
    MMPROJ_FILE=Ternary-Bonsai-2-27B-mmproj-BF16.gguf \
    MMPROJ_ALTERNATE_FILE=Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
    CONTEXT_SIZE=32768 \
    GPU_LAYERS=99 \
    PARALLEL=1 \
    DEFAULT_MAX_TOKENS=8192 \
    DEFAULT_REASONING_EFFORT=medium \
    SERVER_HOST=127.0.0.1 \
    SERVER_PORT=8080 \
    STARTUP_TIMEOUT=900 \
    REQUEST_TIMEOUT=1800

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Copy the built executable and its companion libraries together. Keeping the
# backend plugins beside the executable lets llama.cpp discover them at runtime.
COPY --from=builder /src/llama.cpp/build/bin/ /opt/prism/bin/
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY start_bonsai.sh /app/start_bonsai.sh
COPY handler.py /app/handler.py
RUN chmod +x /app/start_bonsai.sh && mkdir -p /workspace/models

EXPOSE 8080
CMD ["python3", "-u", "/app/handler.py"]
