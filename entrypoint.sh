#!/bin/bash
set -e

# ==============================================================================
# RunPod Serverless & Pod Entrypoint for PrismML Bonsai 2 27B
# Handler loop: runpod.serverless.start({"handler": handler})
# ==============================================================================

echo "=========================================================="
echo " Starting PrismML Bonsai 2 27B Serverless / Pod Container"
echo " Target GPUs: RTX 5090 | RTX 6000 Ada | L40S | A40 | A6000"
echo "=========================================================="

# 1. Inspect Hardware & Auto-Detect GPU VRAM
TOTAL_VRAM_MB=0
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || echo "Unknown GPU")
    TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 || echo 0)
    TOTAL_VRAM_MB=${TOTAL_VRAM_MB%.*}
    echo "[Hardware] Detected GPU: $GPU_NAME ($TOTAL_VRAM_MB MB VRAM)"
else
    echo "[Hardware] Warning: nvidia-smi not found. Defaulting to standard 32GB profile."
    TOTAL_VRAM_MB=32000
fi

# 2. Dynamic VRAM & Context Profiling (256k / 128k / slots / KV quantization)
if [ "$CTX_SIZE" = "auto" ]; then
    if [ "$TOTAL_VRAM_MB" -ge 40000 ]; then
        # 48 GB GPUs (L40S, RTX 6000 Ada, A40, A100): Full 256k context + 6 slots
        CTX_SIZE=261120
        [ "$PARALLEL_SLOTS" = "auto" ] && PARALLEL_SLOTS=6
        [ "$KV_CACHE_TYPE_K" = "auto" ] && KV_CACHE_TYPE_K=q8_0
        [ "$KV_CACHE_TYPE_V" = "auto" ] && KV_CACHE_TYPE_V=q8_0
        echo "[Profile] 48GB Class GPU detected -> Configured 256k context with 6 parallel slots (q8_0 KV cache)"
    elif [ "$TOTAL_VRAM_MB" -ge 28000 ]; then
        # 32 GB GPUs (RTX 5090): 128k context (q8_0) or 256k context (q4_0)
        CTX_SIZE=131072
        [ "$PARALLEL_SLOTS" = "auto" ] && PARALLEL_SLOTS=4
        [ "$KV_CACHE_TYPE_K" = "auto" ] && KV_CACHE_TYPE_K=q8_0
        [ "$KV_CACHE_TYPE_V" = "auto" ] && KV_CACHE_TYPE_V=q8_0
        echo "[Profile] 32GB Class GPU (RTX 5090) detected -> Configured 128k context with 4 parallel slots (q8_0 KV cache)"
    else
        # 24 GB GPUs (RTX 4090 / 3090): 64k context (q4_0)
        CTX_SIZE=65536
        [ "$PARALLEL_SLOTS" = "auto" ] && PARALLEL_SLOTS=2
        [ "$KV_CACHE_TYPE_K" = "auto" ] && KV_CACHE_TYPE_K=q4_0
        [ "$KV_CACHE_TYPE_V" = "auto" ] && KV_CACHE_TYPE_V=q4_0
        echo "[Profile] 24GB Class GPU detected -> Configured 64k context with 2 parallel slots (q4_0 KV cache)"
    fi
else
    # Explicit CTX_SIZE was provided
    [ "$PARALLEL_SLOTS" = "auto" ] && PARALLEL_SLOTS=4
    [ "$KV_CACHE_TYPE_K" = "auto" ] && KV_CACHE_TYPE_K=q8_0
    [ "$KV_CACHE_TYPE_V" = "auto" ] && KV_CACHE_TYPE_V=q8_0
fi

# 3. Model Acquisition (Local volume or HuggingFace auto-download)
mkdir -p "$(dirname "$MODEL_PATH")"

if [ ! -f "$MODEL_PATH" ] && [ -n "$HF_REPO_ID" ] && [ -n "$HF_MODEL_FILE" ]; then
    echo "[Model] Model not found locally. Downloading $HF_MODEL_FILE from Hugging Face ($HF_REPO_ID)..."
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='$HF_REPO_ID', filename='$HF_MODEL_FILE', local_dir='$(dirname "$MODEL_PATH")')"
    if [ -n "$HF_MMPROJ_FILE" ]; then
        echo "[Model] Downloading vision projector $HF_MMPROJ_FILE..."
        python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='$HF_REPO_ID', filename='$HF_MMPROJ_FILE', local_dir='$(dirname "$MMPROJ_PATH")')"
    fi
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "[Error] Model file '$MODEL_PATH' does not exist! Please mount a volume containing your GGUF file or specify HF_REPO_ID and HF_MODEL_FILE."
    exit 1
fi

echo "[Model] Loaded GGUF: $MODEL_PATH"
[ -f "$MMPROJ_PATH" ] && echo "[Model] Loaded Vision Projector: $MMPROJ_PATH"

# 4. Construct llama-server Execution Command
CMD="/usr/local/bin/llama-server \
    -m \"$MODEL_PATH\" \
    --host \"$HOST\" \
    --port \"$PORT\" \
    -ngl \"$N_GPU_LAYERS\" \
    -c \"$CTX_SIZE\" \
    -np \"$PARALLEL_SLOTS\" \
    -b \"$BATCH_SIZE\" \
    -ub \"$UBATCH_SIZE\" \
    -ctk \"$KV_CACHE_TYPE_K\" \
    -ctv \"$KV_CACHE_TYPE_V\" \
    -n -1 \
    --cont-batching \
    --metrics"

if [ -f "$MMPROJ_PATH" ]; then
    CMD="$CMD --mmproj \"$MMPROJ_PATH\""
fi

if [ "$FLASH_ATTENTION" = "true" ]; then
    CMD="$CMD -fa"
fi

echo "[Execution] Starting llama-server with command:"
echo "$CMD"

# 5. Check Execution Mode
if [ "$RUNPOD_MODE" = "http" ] || [ "$RUNPOD_MODE" = "pod" ]; then
    echo "[Mode] Starting direct HTTP llama-server for Pod / API Gateway..."
    exec eval "$CMD"
fi

# Standard Serverless Queue Mode (Default):
echo "[RunPod] Starting llama-server in background..."
eval "$CMD" &
LLAMA_PID=$!

echo "[RunPod] Waiting for llama-server readiness on port $PORT..."
RETRIES=0
MAX_RETRIES=120
until curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 || curl -s "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; do
    sleep 1
    RETRIES=$((RETRIES + 1))
    if [ "$RETRIES" -ge "$MAX_RETRIES" ]; then
        echo "[RunPod] Error: llama-server failed to initialize within $MAX_RETRIES seconds."
        kill "$LLAMA_PID" 2>/dev/null || true
        exit 1
    fi
done

echo "[RunPod] llama-server is healthy and ready for requests!"

# Execute serverless handler (runpod.serverless.start)
if [ $# -gt 0 ]; then
    echo "[RunPod] Executing container command: $@"
    exec "$@"
else
    echo "[RunPod] Launching Python RunPod Serverless queue handler..."
    exec python3 -u /app/handler.py
fi