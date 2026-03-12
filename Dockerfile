FROM python:3.12-slim

WORKDIR /app

# System dependencies (audioop needs gcc on some platforms)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure module __init__ files exist
RUN touch stt/__init__.py tts/__init__.py agent/__init__.py

EXPOSE 8000

# Single worker — scale horizontally with multiple containers
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--log-level", "info"]
