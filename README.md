# Running the LLM Chatbot App in Docker

## 1. Clone the Repository
```bash
git clone <repo-url>
cd <repo-directory>
```

## 2. Start Docker Engine (if needed)
```bash
systemctl --user start docker
```

## 3. Build the Docker Image
```bash
docker build -t llm-chatbot-app .
```

## 4. Prepare a host-visible models cache (one time)
The app and embeddings both use a single shared cache directory on the host:
`~/chatbot_assignment2/models_cache`.

Create it and optionally pre-download the models into it using the container:
```bash
mkdir -p ~/chatbot_assignment2/models_cache
# Optional: pre-populate the cache on the host using the container helper
docker run --rm \
  -e MODELS_CACHE_DIR=/app/models_cache \
  -v ~/chatbot_assignment2/models_cache:/app/models_cache \
  llm-chatbot-app python -u download_models.py
```

Notes:
- The image also downloads the models during the build into an internal cache, and on first start the container copies them into `/app/models_cache` if that directory is empty.
- When you mount the host cache at `/app/models_cache`, the app will use your host’s copies.
- Important: Do not mount an empty host cache if you don’t want seeding; either pre-populate it as above, or let the container seed it on first run.

### Verify the models cache is visible
On the host:
```bash
ls -lah ~/chatbot_assignment2/models_cache
```
Inside the container (optional):
```bash
docker run --rm \
  -e MODELS_CACHE_DIR=/app/models_cache \
  -v ~/chatbot_assignment2/models_cache:/app/models_cache \
  llm-chatbot-app bash -lc 'ls -lah /app/models_cache'
```

## 5. Run the Ingestion Script
Creates the FAISS index for the chatbot.
```bash
docker run --rm --gpus all \
  -e MODELS_CACHE_DIR=/app/models_cache \
  -v $(pwd)/faiss_data:/app/faiss_data \
  -v ~/chatbot_assignment2/models_cache:/app/models_cache \
  llm-chatbot-app python ingest.py
```

## 6. Serve the Chatbot App
```bash
docker run --gpus all -p 5000:5000 \
  -e MODELS_CACHE_DIR=/app/models_cache \
  -v $(pwd)/faiss_data:/app/faiss_data \
  -v ~/chatbot_assignment2/models_cache:/app/models_cache \
  llm-chatbot-app
```

## 7. Access the App via SSH Tunnel (if remote)
- On your local machine, start an SSH tunnel in a new terminal:
  ```bash
  ssh -L 30123:localhost:5000 username@your.remote.host
  ```
- Open your browser and go to:
  ```
  http://localhost:30123
  ```
