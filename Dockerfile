FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/synthetic-lm-pipeline/{data,models,logs,mlruns,reports} \
    && mkdir -p /app/synthetic-lm-pipeline/data/{raw,processed,synthetic,uploads}

# Set permissions
RUN chmod +x /app/scripts/*.sh 2>/dev/null || true

# Expose port for FastAPI
EXPOSE 8000

# Default command
CMD ["python", "src/serve.py", "--host", "0.0.0.0", "--port", "8000"]