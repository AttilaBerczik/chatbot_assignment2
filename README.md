# llm-chatbot-app

## Getting Started

### 1. SSH Port Forwarding

Forward local port `30123` to remote port `5002`:
If the port is occupied, choose another port. Replace all occurrences of `5002` with your chosen port.
```bash
ssh -L 30123:localhost:5002 username@your.remote.host
```

### 2. Build Docker Image

```bash
docker build -t llm-chatbot-app .
```

### 3. Download Models

```bash
docker run --rm \
  -e MODELS_CACHE_DIR=/app/models_cache \
  -v ~/chatbot_assignment2/models_cache:/app/models_cache \
  llm-chatbot-app python -u download_models.py
```

Verify that models are downloaded to `~/chatbot_assignment2/models_cache`.

### 4. Ingest Data

```bash
docker run -it --rm \
  -e HF_HOME=/root/chatbot_assignment2/models_cache \
  -v ~/chatbot_assignment2/models_cache:/root/chatbot_assignment2/models_cache \
  -v $(pwd)/faiss_data:/app/faiss_data \
  llm-chatbot-app python -u ingest.py
```

### 5. Run the App

```bash
docker run --gpus all -p 5000:5000 \
  -e MODELS_CACHE_DIR=/app/models_cache \
  -v "$(pwd)"/faiss_data:/app/faiss_data \
  -v ~/chatbot_assignment2/models_cache:/app/models_cache \
  llm-chatbot-app
```

### 6. Access the App

Open your browser and go to:

```
http://localhost:30123
```