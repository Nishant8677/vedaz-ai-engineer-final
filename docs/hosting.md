# VPS Hosting & Model Deployment Guide with vLLM

This guide outlines the production deployment of the fine-tuned Qwen 3B model for Vedaz using vLLM on a Virtual Private Server (VPS).

## 1. What is vLLM?

vLLM is a highly optimized, high-throughput, and memory-efficient LLM inference engine. 
- **Inference Engine:** It uses PagedAttention to efficiently manage attention keys and values, drastically reducing VRAM fragmentation.
- **Optimized Serving:** Provides continuous batching, which means it can serve multiple concurrent requests much faster than standard HuggingFace pipelines.
- **OpenAI-Compatible API:** vLLM exposes an HTTP server that mimics the OpenAI API schema, allowing easy integration with existing codebases that expect OpenAI endpoints.

## 2. VPS Requirements

To host the Qwen-2.5-3B model efficiently in FP16 or quantized formats, the following VPS specifications are recommended:

*   **OS:** Ubuntu 22.04 LTS
*   **GPU:** 1x NVIDIA RTX A4000, A5000, or A10G (minimum 16GB VRAM for decent context length and batching).
*   **RAM:** 32 GB system memory.
*   **Storage:** 100 GB NVMe SSD (to store the base model, merged model, and Docker images).
*   **CUDA:** Version 12.1+

## 3. Installation Steps

### Step 3.1: Install CUDA and NVIDIA Drivers
Ensure your VPS has the proper NVIDIA drivers installed.
```bash
sudo apt update
sudo apt install nvidia-driver-535 -y
sudo reboot
```

### Step 3.2: Install Python and vLLM
It's recommended to use a virtual environment or Conda.
```bash
# Create and activate virtual environment
python3 -m venv vllm-env
source vllm-env/bin/activate

# Install vLLM
pip install vllm
```

## 4. Loading the Model

Once vLLM is installed and the merged model (Base + LoRA) is transferred to the VPS (e.g., in `/app/models/vedaz-qwen-3b-ft`), you can start the OpenAI-compatible server.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /app/models/vedaz-qwen-3b-ft \
    --served-model-name vedaz-astrologer-v1 \
    --dtype bfloat16 \
    --max-model-len 4096
```

*Note: If VRAM is tight, you can add `--quantization awq` provided the model was exported in AWQ format, or use FP8 kv-cache.*

## 5. API Example

Once the server is running on `http://localhost:8000`, you can query it just like the OpenAI API.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vedaz-astrologer-v1",
    "messages": [
      {"role": "system", "content": "You are Vedaz AI Vedic astrologer. You give compassionate, non-fatalistic guidance."},
      {"role": "user", "content": "Mera career kaisa rahega?"}
    ],
    "temperature": 0.7
  }'
```

## 6. Production Considerations

To make this deployment truly production-ready, consider the following architecture additions:

1.  **Reverse Proxy (NGINX/Caddy):** Do not expose port 8000 directly. Use NGINX to route port 443 (HTTPS) to localhost:8000.
2.  **HTTPS (TLS/SSL):** Use Let's Encrypt / Certbot with NGINX to secure API payloads over the network.
3.  **Authentication:** Add a gateway layer (like Kong or a simple Express.js proxy) to enforce API Keys and rate limiting before requests hit vLLM.
4.  **Monitoring & Logging:** 
    *   Use Prometheus (vLLM natively exposes metrics at `/metrics`).
    *   Set up a Grafana dashboard to track token throughput, request latency, and GPU VRAM usage.
5.  **Process Management:** Use `systemd` or `pm2` to ensure the vLLM server restarts automatically on crashes or server reboots. Alternatively, containerize the application using Docker and manage it via Docker Compose.
