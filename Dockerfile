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

# Try to install flash-attn, but don't fail if it doesn't work
RUN pip install --no-cache-dir flash-attn 2>&1 || echo "⚠️ Flash Attention installation skipped (optional optimization)"

# Set environment variables for a unified models cache
ENV PYTHONUNBUFFERED=1 \
    MODELS_CACHE_DIR=/cache \
    HF_HOME=/cache \
    HUGGINGFACE_HUB_CACHE=/cache

# Create directories
RUN mkdir -p /app/faiss_data /cache

# Copy the application files
COPY . .

# Download models into the shared cache during build
RUN python -u download_models.py

# Expose the port the Flask app runs on
EXPOSE 5000

# Command to run the Flask application with unbuffered output
CMD ["python", "-u", "app.py"]
