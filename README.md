# PrismML Bonsai 2 27B Serverless & Pod Docker Image

A production-ready, multi-architecture CUDA 12.8 container for **PrismML Bonsai 2 27B** with native 2-bit ternary quantization (`PQ2_0`), multimodal vision (`mmproj`), and **up to 256k context**.

---

### **Supported Hardware & Architectures**

Built with `CMAKE_CUDA_ARCHITECTURES="80;86;89;90;100;120"`:
* **Blackwell**: NVIDIA GeForce RTX 5090 (`sm_100` / `sm_120`)
* **Ada Lovelace**: NVIDIA RTX 6000 Ada, L40S, L40, RTX 4090 (`sm_89`)
* **Hopper**: NVIDIA H100 (`sm_90`)
* **Ampere**: NVIDIA A40, A100, RTX A6000, RTX 3090 (`sm_80` / `sm_86`)

---

### **Hardware Auto-Detection Profiles**

The container automatically inspects available GPU VRAM at boot via `nvidia-smi` and tunes the context window, concurrency slots, and KV cache quantization:

| Detected GPU Class | VRAM Range | Context Window | Parallel Slots | KV Cache Type | Typical Cards |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **48GB+ Enterprise** | $\ge 40$ GB | **`261,120` (256k)** | **6** | `q8_0` | L40S, RTX 6000 Ada, A40, A100 |
| **32GB High-Speed** | $28 - 39$ GB | **`131,072` (128k)** *(or 256k with q4)* | **4** | `q8_0` *(or q4_0)* | RTX 5090 |
| **24GB Prosumer** | $< 28$ GB | **`65,536` (64k)** | **2** | `q4_0` | RTX 4090, RTX 3090 |

---

### **Building and Pushing the Docker Image**

```bash
cd docker/bonsai-runpod

# Build with maximum CPU workers (defaults to all cores $(nproc) + NVCC multi-arch parallel threads)
docker build -t your-dockerhub-username/bonsai2-serverless:latest .

# Optional: Specify explicit worker count (e.g. 32 cores on a large build server)
docker build --build-arg MAX_JOBS=32 -t your-dockerhub-username/bonsai2-serverless:latest .

# Push to your registry
docker push your-dockerhub-username/bonsai2-serverless:latest
```

---

### **Deploying on RunPod**

#### **Option A: RunPod Serverless (Recommended)**
1. In the **RunPod Console**, go to **Serverless** -> **New Endpoint**.
2. **Container Image**: `your-dockerhub-username/bonsai2-serverless:latest`.
3. **GPU Types**: Select `L40S 48GB`, `RTX 6000 Ada 48GB`, `A40 48GB`, or `RTX 5090`.
4. **Container Disk**: `35 GB` (or mount a Network Volume with your model at `/workspace/models`).
5. **Max Concurrency per Worker**: `6` (for 48GB) or `4` (for 5090).
6. **Execution Timeout**: `300` seconds.

#### **Option B: RunPod Standard Pod**
1. Select an instance with an **RTX 5090**, **L40S**, or **RTX 6000 Ada**.
2. Specify the image and expose HTTP port `8000`.

---

### **Environment Variable Overrides**

You can customize runtime behavior by adding these environment variables in your RunPod template:

| Variable | Default | Description |
| :--- | :---: | :--- |
| `MODEL_PATH` | `/workspace/models/Ternary-Bonsai-2-27B-PQ2_0.gguf` | Absolute path to GGUF model file. |
| `MMPROJ_PATH` | `/workspace/models/mmproj-bonsai-2-f16.gguf` | Absolute path to vision projector. |
| `HF_REPO_ID` | `""` | Optional: HuggingFace repository to auto-download from. |
| `HF_MODEL_FILE` | `""` | Optional: GGUF filename on HuggingFace. |
| `HF_MMPROJ_FILE` | `""` | Optional: Projector filename on HuggingFace. |
| `CTX_SIZE` | `auto` | Override context size (e.g. `261120`, `131072`, `65536`). |
| `PARALLEL_SLOTS` | `auto` | Override concurrency slots (e.g. `4` or `6`). |
| `KV_CACHE_TYPE_K` | `auto` | `q8_0` or `q4_0` (use `q4_0` for 256k context on RTX 5090). |
| `KV_CACHE_TYPE_V` | `auto` | `q8_0` or `q4_0`. |
| `FLASH_ATTENTION` | `true` | Must be `true` for maximum throughput and low VRAM usage. |
| `MAX_TOKENS` | `8192` | Unlimited output tokens so client requests dictate generation limits. |

---

### **Connecting to Framed Video Bot Framework**

In Framed Studio -> **Model Config**:
* **Provider**: `openai-compatible`
* **Base URL**: `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1`
* **API Key**: `rpa_...`
* **Ingest Concurrency**: `6` (for 48GB cards) or `4` (for RTX 5090)
* **Context Tier**: `256k`
* **Text Walk Window Size**: `7`
* **Enable Thinking**: `OFF` / **Reasoning Effort**: `none`
