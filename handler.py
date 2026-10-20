"""Queue adapter for the PrismML Bonsai 2 multimodal llama.cpp server."""
import atexit
import os
import subprocess
import time
from typing import Any

import requests
import runpod

SERVER_HOST = "127.0.0.1"
SERVER_PORT = int(os.getenv("SERVER_PORT", "8080"))
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
STARTUP_TIMEOUT = int(os.getenv("STARTUP_TIMEOUT", "900"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "1800"))
_server: subprocess.Popen | None = None


def _start_server() -> None:
    global _server
    env = os.environ.copy()
    if "MODEL_DIR" not in env:
        if os.path.isdir("/runpod-volume"):
            env["MODEL_DIR"] = "/runpod-volume/models/bonsai2"
        else:
            env["MODEL_DIR"] = "/workspace/models"

    _server = subprocess.Popen(
        ["/app/start_bonsai.sh"],
        env=env,
        stdout=None,
        stderr=None,
    )
    deadline = time.monotonic() + STARTUP_TIMEOUT
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        if _server.poll() is not None:
            raise RuntimeError(
                f"llama-server exited during startup with code {_server.returncode}"
            )
        try:
            response = requests.get(f"{SERVER_URL}/health", timeout=5)
            if response.ok:
                print("[worker] Multimodal Bonsai 2 server is ready", flush=True)
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(2)

    _server.terminate()
    raise RuntimeError(
        f"Timed out waiting for llama-server after {STARTUP_TIMEOUT}s: {last_error}"
    )


def _stop_server() -> None:
    if _server is not None and _server.poll() is None:
        _server.terminate()


atexit.register(_stop_server)
_start_server()


def _normalize_payload(job_input: dict[str, Any]) -> dict[str, Any]:
    # Runpod's OpenAI Gateway wrapper and ordinary queue payloads are both accepted.
    payload = job_input.get("openai_input") or job_input.get("payload") or job_input.get("request")
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(job_input.get("messages"), list):
        payload = dict(job_input)
    elif "prompt" in job_input:
        payload = {"messages": [{"role": "user", "content": job_input["prompt"]}]}
        for key in (
            "temperature", "top_p", "top_k", "min_p", "max_tokens",
            "reasoning_effort", "tools", "tool_choice", "response_format",
            "seed", "stop",
        ):
            if key in job_input:
                payload[key] = job_input[key]
    else:
        raise ValueError("input must contain 'messages', 'prompt', or 'payload'/'request'")

    payload.setdefault("max_tokens", int(os.getenv("DEFAULT_MAX_TOKENS", "8192")))
    payload.setdefault("temperature", 1.0)
    payload.setdefault("top_p", 0.95)
    payload.setdefault("top_k", 20)
    payload.setdefault("min_p", 0.05)
    payload.setdefault("reasoning_effort", os.getenv("DEFAULT_REASONING_EFFORT", "medium"))
    return payload


def handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        job_input = job.get("input") or {}
        if not isinstance(job_input, dict):
            return {"error": "Runpod input must be a JSON object"}

        route = str(job_input.get("openai_route") or "/v1/chat/completions")
        if not route.startswith("/"):
            route = "/" + route
        if route.startswith("/openai/"):
            route = route[len("/openai"):]
        if not route.startswith("/v1/"):
            route = "/v1" + route
        if route.endswith("/models"):
            response = requests.get(f"{SERVER_URL}{route}", timeout=30)
        else:
            payload = _normalize_payload(job_input)
            payload.setdefault("model", os.getenv("MODEL_FILE", "Ternary-Bonsai-2-27B-PQ2_0.gguf"))
            # Keep image_url/content-part arrays intact; llama-server consumes the
            # standard OpenAI multimodal message format when --mmproj is enabled.
            response = requests.post(
                f"{SERVER_URL}{route}", json=payload, timeout=REQUEST_TIMEOUT
            )

        try:
            body = response.json()
        except ValueError:
            body = {"error": response.text}
        if not response.ok:
            return {
                "error": "llama-server request failed",
                "status_code": response.status_code,
                "detail": body,
            }
        return body
    except Exception as exc:
        return {"error": type(exc).__name__, "detail": str(exc)}


runpod.serverless.start({"handler": handler})
