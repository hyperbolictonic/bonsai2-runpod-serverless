"""
RunPod Serverless Queue Handler for Bonsai 2 27B
Proxies RunPod serverless job payloads into the local llama-server OpenAI endpoint.
"""
import os
import runpod
import aiohttp
import asyncio

LOCAL_OPENAI_URL = f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/v1/chat/completions"

async def handler(job):
    job_input = job.get("input", {})
    
    # Extract standard OpenAI chat-completions payload
    messages = job_input.get("messages", [])
    if not messages and "prompt" in job_input:
        messages = [{"role": "user", "content": job_input["prompt"]}]
        
    payload = {
        "messages": messages,
        "temperature": job_input.get("temperature", 0.0),
        "max_tokens": job_input.get("max_tokens", 8192),
        "stream": False
    }
    
    if "tools" in job_input:
        payload["tools"] = job_input["tools"]
    if "tool_choice" in job_input:
        payload["tool_choice"] = job_input["tool_choice"]
    if "response_format" in job_input:
        payload["response_format"] = job_input["response_format"]

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(LOCAL_OPENAI_URL, json=payload, timeout=300) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data
                else:
                    err_text = await resp.text()
                    return {"error": f"llama-server returned HTTP {resp.status}", "details": err_text}
        except Exception as e:
            return {"error": f"Failed to forward request to llama-server: {str(e)}"}

if __name__ == "__main__":
    print("[RunPod Handler] Initializing RunPod Serverless worker...")
    runpod.serverless.start({"handler": handler})
