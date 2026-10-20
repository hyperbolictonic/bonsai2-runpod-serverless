# Bonsai 2 27B multimodal on Runpod Serverless

Queue-based worker serving PrismML Bonsai 2 through the PrismML llama.cpp fork. The worker uses the PQ2_0 ternary language GGUF and downloads both separate vision projectors. BF16 (about 888 MiB) is selected by default; Q8_0 is retained as an alternative. llama-server loads only the selected projector, so the endpoint accepts text-only and image-containing chat messages.

The build/runtime is adapted from the working [Hexadi Runpod worker](https://github.com/hexadi/bonsai2-runpod-serverless), with the optional vision projector added. The PrismML llama.cpp source is pinned to commit `73bfe6cc485554822f80c473a3c0c4f25e80f41b` for repeatable builds.

## Deploy

In Runpod Serverless, choose **Add your repo**, select this repository and `main`, and use the root `Dockerfile`. The image uses CUDA 12.8 and compiles native CUDA targets for Ampere `sm_80` and consumer Blackwell `sm_120`; the embedded forward-compatible PTX provides a fallback for intermediate architectures (including `sm_86/89/90`) and Blackwell `sm_100`. CUDA 12.8 is required for Blackwell. Flash Attention is explicitly enabled at server startup; actual compatibility still depends on the GPU/backend and the pinned PrismML llama.cpp revision, so verify on each target class before production.

The model and both projector variants download at worker startup; they are not baked into the image. Allow at least 20 GB of worker/container disk for the model, both projectors, Hugging Face metadata/cache, and temporary working space. If you attach a Runpod network volume at `/runpod-volume`, the worker reuses `/runpod-volume/models/bonsai2` across worker replacements; otherwise files go under `/workspace/models` on the worker disk.

PrismML publishes both the BF16 reference projector (931 MB decimal, about 888 MiB) and the smaller Q8_0 projector (629 MB). BF16 is the default active projector; set `MMPROJ_FILE=Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` to use Q8_0 instead. Both files remain on disk. See the [model files and documentation](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf); PQ2_0 model weights are about 7.21 GB.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_REPO` | `prism-ml/Ternary-Bonsai-2-27B-gguf` | Hugging Face repository |
| `MODEL_FILE` | `Ternary-Bonsai-2-27B-PQ2_0.gguf` | Language model file |
| `MMPROJ_FILE` | `Ternary-Bonsai-2-27B-mmproj-BF16.gguf` | Active projector; set to the Q8_0 filename to switch, or empty to disable image input |
| `MMPROJ_ALTERNATE_FILE` | `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | Second projector downloaded for easy switching |
| `MODEL_DIR` | auto: volume if mounted, otherwise `/workspace/models` | Persistent model directory |
| `HF_TOKEN` | unset | Optional Hugging Face token |
| `CONTEXT_SIZE` | `32768` | Context tokens; raise only after checking GPU memory |
| `GPU_LAYERS` | `99` | Layers to offload to GPU |
| `BATCH_SIZE` | `512` | Logical batch size (`-b`) |
| `UBATCH_SIZE` | `512` | Physical batch size (`-ub`) |
| KV cache | `q4_0` for K and V | Quantized KV cache to reduce memory use |
| Flash Attention | `on` | Enabled for supported GPU/build combinations |
| `PARALLEL` | `1` | llama-server parallel slots |
| `STARTUP_TIMEOUT` | `900` | Time allowed for downloads and model startup |
| `REQUEST_TIMEOUT` | `1800` | Per-request timeout in seconds |

## Text request

Send to the endpoint's `/run` route:

```json
{
  "input": {
    "messages": [
      {"role": "user", "content": "Explain how a solar panel works."}
    ],
    "max_tokens": 1024
  }
}
```

## Image request

Use the OpenAI multimodal `content` array. A data URL avoids depending on the worker being able to fetch an external image URL; replace the placeholder with a base64-encoded JPEG or PNG.

```json
{
  "input": {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this image."},
          {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,<BASE64_IMAGE>"}
          }
        ]
      }
    ],
    "max_tokens": 1024
  }
}
```

The queue adapter passes the message content through to llama-server without rewriting image parts. Both projector files are downloaded, but only the selected one is loaded into the server and uses GPU memory.

## Build locally

```sh
docker build -t bonsai2-runpod-serverless:multimodal .
```

The CUDA build can take several minutes and requires enough temporary build disk. The HF model files are downloaded only when a worker starts.
