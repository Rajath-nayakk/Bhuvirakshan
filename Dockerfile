# Forest & River Guardian - Production Container
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies for GDAL, GEOS, OpenCV, Libsndfile, and SQLite
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, configs, frontend, and offline sample data
COPY configs/ configs/
COPY src/ src/
COPY frontend/ frontend/
COPY data/samples/ data/samples/
COPY reports/ reports/
COPY .env.example .env

# Expose FastAPI and Leaflet Web Dashboard port
EXPOSE 8000

# Healthcheck endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/docs || exit 1

# Start FastAPI server on port 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
