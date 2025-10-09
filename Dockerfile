# syntax=docker/dockerfile:1.4

# Use the official Python image as a base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preload Hugging Face models and embeddings into cache with BuildKit cache mounts
ENV HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_TIMEOUT=300

# Copy preload script and execute it
COPY preload.py .
RUN --mount=type=cache,id=huggingface-cache,target=/root/.cache/huggingface \
    --mount=type=cache,id=torch-cache,target=/root/.cache/torch \
    python preload.py

# Copy the rest of the application files
COPY . .

# Expose the port the Flask app runs on
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "app.py"]
