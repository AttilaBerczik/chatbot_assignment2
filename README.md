# Running the LLM Chatbot App in Docker

## 1. Clone the Repository
```bash
git clone <repo-url>
cd <repo-directory>
```

## 2. Set Up Docker (Rootless)
- View setup instructions:
  ```bash
  cat /home/shared/docker-setup.sh
  ```
- Follow the script to configure Docker for your user.

## 3. Start Docker Engine
```bash
systemctl --user start docker
```

## 4. Build the Docker Image
```bash
docker build -t llm-chatbot-app .
```

## 5. Prepare a host-visible models cache (one time)
The app and embeddings both use a single shared cache directory on the host:
`~/chatbot_assignment2/models_cache`.

Create it and optionally pre-download the models into it using the container:
```bash
mkdir -p ~/chatbot_assignment2/models_cache
# Optional: pre-populate the cache on the host using the container helper
docker run --rm \
  -e MODELS_CACHE_DIR=/cache \
  -v ~/chatbot_assignment2/models_cache:/cache \
  llm-chatbot-app python -u download_models.py
```

Notes:
- The image also downloads the models during the build into `/cache` for faster cold starts.
- When you mount the host cache at `/cache`, it takes precedence so models are read from your host.
- Important: Do not mount an empty host cache, as it will hide the models baked into the image. Either pre-populate it as above, or omit the `-v ~/...:/cache` mount and the app will use the baked-in `/cache`.

## 6. Run the Ingestion Script
Creates the FAISS index for the chatbot.
```bash
docker run --rm --gpus all \
  -e MODELS_CACHE_DIR=/cache \
  -v $(pwd)/faiss_data:/app/faiss_data \
  -v ~/chatbot_assignment2/models_cache:/cache \
  llm-chatbot-app python ingest.py
```

## 7. Serve the Chatbot App
```bash
docker run --gpus all -p 5000:5000 \
  -e MODELS_CACHE_DIR=/cache \
  -v $(pwd)/faiss_data:/app/faiss_data \
  -v ~/chatbot_assignment2/models_cache:/cache \
  llm-chatbot-app
```

## 8. Access the App via SSH Tunnel
- On your local machine, start an SSH tunnel in a new terminal:
  ```bash
  ssh -L 30123:localhost:5000 username@vs-c1.cs.uit.no
  ```
- Open your browser and go to:
  ```
  http://localhost:30123
  ```
