"""
RunPod Serverless OpenAI-Compatible Queue Handler for PrismML Bonsai 2 27B
Proxies both RunPod OpenAI Gateway routes (/openai/v1/models, /openai/v1/chat/completions)
and standard serverless job payloads into the local llama-server OpenAI backend.
"""
import os
import sys
import time
import subprocess
import requests
import runpod

LOCAL_PORT = os.environ.get("PORT", "8000")
BASE_URL = f"http://127.0.0.1:{LOCAL_PORT}"
HEALTH_URL = f"{BASE_URL}/health"
MODELS_URL = f"{BASE_URL}/v1/models"
CHAT_URL = f"{BASE_URL}/v1/chat/completions"

def is_server_ready():
    try:
        r = requests.get(HEALTH_URL, timeout=1)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    try:
        r = requests.get(MODELS_URL, timeout=1)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    return False

def ensure_llama_server():
    """Ensure llama-server backend is running and healthy."""
    if is_server_ready():
        print("[RunPod Handler] llama-server backend is already running and ready.")
        return

    entrypoint_script = "/app/entrypoint.sh" if os.path.exists("/app/entrypoint.sh") else "./entrypoint.sh"
    if os.path.exists(entrypoint_script):
        print(f"[RunPod Handler] Spawning llama-server via {entrypoint_script} in background...")
        env = os.environ.copy()
        env["RUNPOD_MODE"] = "http"
        subprocess.Popen(["bash", entrypoint_script], env=env)
    else:
        cmd = [
            "/usr/local/bin/llama-server",
            "-m", os.environ.get("MODEL_PATH", "/workspace/models/Ternary-Bonsai-2-27B-PQ2_0.gguf"),
            "--host", os.environ.get("HOST", "0.0.0.0"),
            "--port", str(LOCAL_PORT),
            "-ngl", os.environ.get("N_GPU_LAYERS", "99"),
            "-c", os.environ.get("CTX_SIZE", "261120"),
            "-np", os.environ.get("PARALLEL_SLOTS", "6"),
            "-fa"
        ]
        print(f"[RunPod Handler] Spawning direct llama-server: {' '.join(cmd)}")
        subprocess.Popen(cmd)

    print(f"[RunPod Handler] Waiting for llama-server readiness on port {LOCAL_PORT} (up to 600s)...")
    for i in range(1, 601):
        if is_server_ready():
            print(f"[RunPod Handler] llama-server backend is healthy and ready after {i}s!")
            return
        if i % 15 == 0:
            print(f"[RunPod Handler] Still waiting for llama-server readiness ({i}s elapsed)...")
        time.sleep(1)

    print("[RunPod Handler] Warning: llama-server health check timed out. Proceeding to start handler.")

def handler(job):
    """
    Standard RunPod Serverless queue handler function.
    Bridges RunPod OpenAI Gateway calls and raw jobs into llama-server.
    """
    job_input = job.get("input", {}) if isinstance(job, dict) else {}
    if not job_input:
        return {"error": "Empty input payload provided."}

    # 1. Determine the target route (RunPod OpenAI Gateway passes "openai_route")
    raw_route = str(job_input.get("openai_route", "")).strip()
    if not raw_route:
        raw_route = "/v1/chat/completions"

    if not raw_route.startswith("/"):
        raw_route = "/" + raw_route
    if not raw_route.startswith("/v1"):
        raw_route = "/v1" + raw_route

    target_url = f"{BASE_URL}{raw_route}"

    # 2. Extract payload (RunPod OpenAI Gateway passes "openai_input")
    openai_input = job_input.get("openai_input")
    if isinstance(openai_input, dict):
        payload = openai_input
    else:
        payload = dict(job_input)
        payload.pop("openai_route", None)
        payload.pop("openai_input", None)

    # 3. GET routes (like /v1/models)
    if raw_route.endswith("/models") or (not payload and "chat" not in raw_route):
        try:
            resp = requests.get(target_url, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            # Fallback list if llama-server /v1/models is initializing
            model_name = os.environ.get("HF_MODEL_FILE", "Ternary-Bonsai-2-27B-PQ2_0.gguf").replace(".gguf", "")
            return {
                "object": "list",
                "data": [
                    {
                        "id": model_name,
                        "object": "model",
                        "owned_by": "prism-ml"
                    }
                ]
            }
        except Exception as exc:
            return {"error": f"Failed to query {target_url}: {str(exc)}"}

    # 4. POST routes (like /v1/chat/completions or /v1/completions)
    if "messages" not in payload and "prompt" in payload:
        payload["messages"] = [{"role": "user", "content": payload.pop("prompt")}]

    if "model" not in payload:
        payload["model"] = os.environ.get("HF_MODEL_FILE", "Ternary-Bonsai-2-27B-PQ2_0")

    timeout_sec = job_input.get("timeout", 600)
    try:
        resp = requests.post(target_url, json=payload, timeout=timeout_sec)
        if resp.status_code == 200:
            return resp.json()
        return {
            "error": f"llama-server returned HTTP {resp.status_code}",
            "details": resp.text
        }
    except Exception as exc:
        return {"error": f"Failed to forward request to llama-server at {target_url}: {str(exc)}"}

ensure_llama_server()
print("[RunPod Handler] Initializing RunPod Serverless queue worker loop...")
runpod.serverless.start({"handler": handler})