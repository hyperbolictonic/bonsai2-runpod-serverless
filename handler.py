"""
RunPod Serverless Queue Handler for PrismML Bonsai 2 27B
Proxies RunPod serverless job payloads into the local llama-server OpenAI endpoint.
"""
import os
import json
import time
import requests
import runpod

LOCAL_OPENAI_URL = f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/v1/chat/completions"

def handler(job):
    """
    Standard synchronous RunPod Serverless queue handler function.
    Receives incoming job event and forwards to local llama-server.
    """
    # Accept both standard dict and nested event schemas
    job_input = job.get("input", {}) if isinstance(job, dict) else {}
    if not job_input:
        return {"error": "Empty input payload provided."}

    # Extract messages or raw prompt
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

    # Pass optional OpenAI parameters if provided
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
    print("[RunPod Handler] Initializing RunPod Serverless queue worker loop...")
    runpod.serverless.start({"handler": handler})
