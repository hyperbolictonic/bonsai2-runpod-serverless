"""
RunPod Serverless Queue Handler for PrismML Bonsai 2 27B
Proxies RunPod serverless job payloads into the local llama-server OpenAI endpoint.
"""
import os
import sys
import time
import subprocess
import requests
import runpod

LOCAL_PORT = os.environ.get("PORT", "8000")
LOCAL_OPENAI_URL = f"http://127.0.0.1:{LOCAL_PORT}/v1/chat/completions"
HEALTH_URL = f"http://127.0.0.1:{LOCAL_PORT}/health"
MODELS_URL = f"http://127.0.0.1:{LOCAL_PORT}/v1/models"

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

    # Check if entrypoint.sh exists to launch configured llama-server
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
            "-c", os.environ.get("CTX_SIZE", "131072"),
            "-np", os.environ.get("PARALLEL_SLOTS", "4"),
            "-fa"
        ]
        print(f"[RunPod Handler] Spawning direct llama-server: {' '.join(cmd)}")
        subprocess.Popen(cmd)

    print(f"[RunPod Handler] Waiting for llama-server readiness on port {LOCAL_PORT}...")
    for _ in range(120):
        if is_server_ready():
            print("[RunPod Handler] llama-server backend is healthy and ready!")
            return
        time.sleep(1)

    print("[RunPod Handler] Warning: llama-server health check timed out. Proceeding to start handler.")

def handler(job):
    """
    Standard synchronous RunPod Serverless queue handler function.
    Receives incoming job event and forwards to local llama-server.
    """
    job_input = job.get("input", {}) if isinstance(job, dict) else {}
    if not job_input:
        return {"error": "Empty input payload provided."}

    messages = job_input.get("messages", [])
    if not messages and "prompt" in job_input:
        messages = [{"role": "user", "content": job_input["prompt"]}]

    if not messages:
        return {"error": "No 'messages' or 'prompt' found in input."}

    payload = {
        "messages": messages,
        "temperature": job_input.get("temperature", 0.0),
        "max_tokens": job_input.get("max_tokens", 8192),
        "stream": False
    }

    for key in ("tools", "tool_choice", "response_format", "top_p", "presence_penalty", "frequency_penalty", "stop"):
        if key in job_input:
            payload[key] = job_input[key]

    timeout_sec = job_input.get("timeout", 600)
    try:
        response = requests.post(LOCAL_OPENAI_URL, json=payload, timeout=timeout_sec)
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"llama-server returned HTTP {response.status_code}",
                "details": response.text
            }
    except Exception as exc:
        return {"error": f"Failed to forward request to llama-server: {str(exc)}"}

if __name__ == "__main__":
    ensure_llama_server()
    print("[RunPod Handler] Initializing RunPod Serverless queue worker loop...")
    runpod.serverless.start({"handler": handler})