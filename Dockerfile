# syntax=docker/dockerfile:1.4

# Use NVIDIA CUDA base image for GPU support and Flash Attention compilation
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install Python and build dependencies required for Flash Attention
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install PyTorch with CUDA support first (required for Flash Attention)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Flash Attention (requires CUDA and build tools - takes time to compile)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for model download
ENV HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_TIMEOUT=300 \
    HF_MODELS_DIR=/app/models

# Copy download script and run it to download models during build
COPY download_models.py .
RUN python download_models.py

# Copy the rest of the application files
COPY . .

# Expose the port the Flask app runs on
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "app.py"]
