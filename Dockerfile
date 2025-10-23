# Use official Python image as base
FROM python:3.10

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir flash-attn 2>&1 || echo "⚠️ Flash Attention install skipped (optional)"

# Set model cache path (shared with host)
ENV HF_HOME=/root/chatbot_assignment2/models_cache \
    HF_HUB_DISABLE_PROGRESS_BARS=0 \
    HF_HUB_TIMEOUT=600 \
    PYTHONUNBUFFERED=1

# Create directories
RUN mkdir -p /root/chatbot_assignment2/models_cache /app/faiss_data

# Copy app
COPY . .

EXPOSE 5000
CMD ["python", "-u", "app.py"]
