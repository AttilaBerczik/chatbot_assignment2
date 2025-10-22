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

## 5. Run the Ingestion Script
Creates the FAISS index for the chatbot.
```bash
docker run --rm --gpus all -v $(pwd)/faiss_data:/app/faiss_data llm-chatbot-app python ingest.py
```

## 6. Serve the Chatbot App
```bash
docker run --gpus all -p 5000:5000 -v $(pwd)/faiss_data:/app/faiss_data llm-chatbot-app
```

## 7. Access the App via SSH Tunnel
- On your local machine, start an SSH tunnel in a new terminal:
  ```bash
  ssh -L 30123:localhost:5000 username@vs-c1.cs.uit.no
  ```
- Open your browser and go to:
  ```
  http://localhost:30123
  ```