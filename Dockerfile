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

# Create directories
RUN mkdir -p /app/faiss_data /app/models_cache

# Copy the rest of the application files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    MODELS_CACHE_DIR=/app/models_cache

# Expose the port the Flask app runs on
EXPOSE 5002