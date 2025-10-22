# syntax=docker/dockerfile:1.4

# Use the official Python image as a base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
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
