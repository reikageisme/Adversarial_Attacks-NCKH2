# Use python:3.9-slim for smaller image size on Pi
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# 1. Install System Dependencies (OpenCV requires these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy Requirements
COPY requirements.txt .

# 3. Install Python Dependencies
# --no-cache-dir to keep image small
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Source Code
COPY src/ ./src/
COPY templates/ ./templates/
COPY app.py .

# 5. Environment Variables
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1

# 6. Expose Port
EXPOSE 5000

# 7. Entry Point
CMD ["python", "app.py"]
