#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-prism-ml/Ternary-Bonsai-2-27B-gguf}"
MODEL_FILE="${MODEL_FILE:-Ternary-Bonsai-2-27B-PQ2_0.gguf}"
MMPROJ_FILE="${MMPROJ_FILE-Ternary-Bonsai-2-27B-mmproj-BF16.gguf}"
MMPROJ_ALTERNATE_FILE="${MMPROJ_ALTERNATE_FILE:-Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf}"
MODEL_DIR="${MODEL_DIR:-/workspace/models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/${MODEL_FILE}}"
MMPROJ_PATH="${MMPROJ_PATH:-${MODEL_DIR}/${MMPROJ_FILE}}"
MMPROJ_ALT_PATH="${MMPROJ_ALT_PATH:-${MODEL_DIR}/${MMPROJ_ALTERNATE_FILE}}"

CONTEXT_SIZE="${CONTEXT_SIZE:-32768}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
GPU_LAYERS="${GPU_LAYERS:-99}"
PARALLEL="${PARALLEL:-1}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8080}"

mkdir -p "${MODEL_DIR}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "${HF_HUB_CACHE}" "${HF_XET_CACHE}" "${TMPDIR}"

export MODEL_REPO MODEL_FILE MMPROJ_FILE MMPROJ_ALTERNATE_FILE MODEL_DIR MODEL_PATH MMPROJ_PATH MMPROJ_ALT_PATH
export HF_TOKEN="${HF_TOKEN:-}"
python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = os.environ["MODEL_REPO"]
directory = os.environ["MODEL_DIR"]
token = os.environ.get("HF_TOKEN") or None

files = [("MODEL_FILE", "MODEL_PATH", "language model")]
if os.environ.get("MMPROJ_FILE", "").strip():
    files.extend([
        ("MMPROJ_FILE", "MMPROJ_PATH", "active vision projector"),
        ("MMPROJ_ALTERNATE_FILE", "MMPROJ_ALT_PATH", "alternate vision projector"),
    ])

for env_name, path_env, label in files:
    filename = os.environ.get(env_name, "").strip()
    if not filename:
        continue
    target = Path(os.environ[path_env])
    if target.is_file() and target.stat().st_size > 0:
        print(f"[model] Reusing {label}: {target}", flush=True)
        continue
    print(f"[model] Downloading {label} {repo}/{filename} to {target.parent}", flush=True)
    downloaded = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=directory,
        token=token,
    )
    if Path(downloaded).stat().st_size == 0:
        raise RuntimeError(f"Downloaded {label} is empty: {downloaded}")
    if Path(downloaded) != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(Path(downloaded))
PY

if [[ ! -s "${MODEL_PATH}" ]]; then
  echo "[error] Language model missing or empty: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ -n "${MMPROJ_FILE}" && ! -s "${MMPROJ_PATH}" ]]; then
  echo "[error] Vision projector missing or empty: ${MMPROJ_PATH}" >&2
  exit 1
fi
if [[ -n "${MMPROJ_FILE}" && -n "${MMPROJ_ALTERNATE_FILE}" && ! -s "${MMPROJ_ALT_PATH}" ]]; then
  echo "[error] Alternate vision projector missing or empty: ${MMPROJ_ALT_PATH}" >&2
  exit 1
fi

export LD_LIBRARY_PATH="/opt/prism/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

LLAMA_CMD=(
  /opt/prism/bin/llama-server
  -m "${MODEL_PATH}"
  --host "${SERVER_HOST}"
  --port "${SERVER_PORT}"
  -ngl "${GPU_LAYERS}"
  -c "${CONTEXT_SIZE}"
  -b "${BATCH_SIZE}"
  -ub "${UBATCH_SIZE}"
  -np "${PARALLEL}"
  -ctk q4_0
  -ctv q4_0
  -fa on
  --jinja
  --temp 1.0
  --top-p 0.95
  --top-k 20
  --min-p 0.05
)

if [[ -n "${MMPROJ_FILE}" ]]; then
  LLAMA_CMD+=(--mmproj "${MMPROJ_PATH}")
fi

echo "[bonsai] Starting multimodal llama-server (model=${MODEL_PATH}, mmproj=${MMPROJ_PATH:-disabled})"
exec "${LLAMA_CMD[@]}"
