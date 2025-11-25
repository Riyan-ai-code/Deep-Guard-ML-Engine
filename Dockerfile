# Use a lightweight Python image
FROM python:3.10-slim

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install necessary system dependencies (like ffmpeg for video processing)
# The '-y' flag assumes yes, and the cleanup lines reduce image size
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Copy the requirements file and install Python dependencies
# This layer only rebuilds if requirements.txt changes (good caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (e.g., the 'app' directory)
COPY . .

# Expose the application port (must match your app's binding)
EXPOSE 8000

# Command to run the application (assuming you use Uvicorn/Gunicorn/etc. with a main app file)
# Example command assuming a file named 'main.py' and an app object 'app'
# You need to adjust this CMD based on your actual ML framework/entrypoint
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]