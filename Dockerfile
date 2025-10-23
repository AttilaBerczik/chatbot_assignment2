# ================================
# 🐍 Base Image
# ================================
FROM python:3.10

# ================================
# 🧩 System Dependencies
# ================================
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ================================
# 📦 Python Dependencies
# ================================
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Optional: performance extras
RUN pip install --no-cache-dir flash-attn 2>&1 || echo "⚠️ Flash Attention install skipped (optional)"
RUN pip install --upgrade --no-cache-dir langchain langchain-core langchain-community langchain-huggingface
RUN pip install --no-cache-dir huggingface_hub

# ================================
# ⚙️ Environment Variables
# ================================
ENV HF_HOME=/root/chatbot_assignment2/models_cache \
    HF_HUB_DISABLE_PROGRESS_BARS=0 \
    HF_HUB_TIMEOUT=600 \
    PYTHONUNBUFFERED=1

# Create required directories
RUN mkdir -p /root/chatbot_assignment2/models_cache /app/faiss_data

# ================================
# 📥 Pre-download Models
# ================================
# Use huggingface-cli instead of python -m
RUN huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir /root/chatbot_assignment2/models_cache/Qwen2.5-7B-Instruct && \
    huggingface-cli download BAAI/bge-large-en-v1.5 \
    --local-dir /root/chatbot_assignment2/models_cache/bge-large-en-v1.5

# ================================
# 🧠 Copy Application
# ================================
COPY . .

EXPOSE 5000

# ================================
# 🚀 Launch App
# ================================
CMD ["python", "-u", "app.py"]
