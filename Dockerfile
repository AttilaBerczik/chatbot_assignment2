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

# Set environment variables for model download
ENV HF_HUB_DISABLE_PROGRESS_BARS=0 \
    HF_HUB_TIMEOUT=600 \
    HF_MODELS_DIR=/app/models \
    PYTHONUNBUFFERED=1

# Create models directory
RUN mkdir -p /app/models /app/faiss_data

# Copy download script
COPY download_models.py .

# Download models with verbose output
# Step 10/13: Download models AND clean up the temporary cache in the same layer
RUN echo "Starting model downloads..." && \
    python download_models.py && \
    echo "Model downloads completed" && \
    rm -rf /root/.cache/huggingface

# Copy the rest of the application files
COPY . .

# Expose the port the Flask app runs on
EXPOSE 5000

# Command to run the Flask application with unbuffered output
CMD ["python", "-u", "app.py"]
