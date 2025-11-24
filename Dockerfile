FROM python:3.10-slim

WORKDIR /app

# Install system deps needed for some packages (opencv, ffmpeg if needed)
RUN apt-get update && \
    apt-get install -y build-essential ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
# FIX: The source path now starts from the repository root (e.g., Deep-Guard-ML-Engine/)
COPY Deep-Guard-ML-Engine/requirements.txt ./ 
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
# FIX: The source path now starts from the repository root (e.g., Deep-Guard-ML-Engine/app)
COPY Deep-Guard-ML-Engine/app ./app

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]