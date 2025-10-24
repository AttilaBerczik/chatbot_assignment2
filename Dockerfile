# Use official Python image as base (more reliable than CUDA)
FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


RUN pip install langchain-classic

# Try to install flash-attn, but don't fail if it doesn't work
RUN pip install --no-cache-dir flash-attn 2>&1 || echo "⚠️ Flash Attention installation skipped (optional optimization)"

# Create directories
RUN mkdir -p /app/faiss_data /app/models_cache

# --- Download models in a cacheable layer (only re-runs if the script or requirements change) ---
#COPY download_models.py ./download_models.py
#RUN mkdir -p /image_models_cache && \
 #   MODELS_CACHE_DIR=/image_models_cache python -u download_models.py

# Copy the rest of the application files (won't invalidate the model layer)
COPY . .

# Set environment variables for a unified runtime models cache under /app/models_cache
ENV PYTHONUNBUFFERED=1 \
    MODELS_CACHE_DIR=/app/models_cache \
    HF_HOME=/app/models_cache \
    HUGGINGFACE_HUB_CACHE=/app/models_cache \
    TRANSFORMERS_CACHE=/app/models_cache

# Ensure entrypoint is executable
RUN chmod +x /app/entrypoint.sh

# Expose the port the Flask app runs on
EXPOSE 5002

# Use the entrypoint to handle cache seeding, then launch the app
ENTRYPOINT ["/app/entrypoint.sh"]
