FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY setup.py .
RUN pip install -e .

# Copy source code
COPY . .

# Expose port for API
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.1"
ENV QUANTIZE="True"

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api:app"] 