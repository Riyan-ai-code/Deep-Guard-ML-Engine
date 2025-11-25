FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y build-essential ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY Deep-Guard-ML-Engine/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY Deep-Guard-ML-Engine/app ./app

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Use Render's assigned port if available, fallback to 8000 locally
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
