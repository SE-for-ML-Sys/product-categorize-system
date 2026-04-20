# ── app/backend/Dockerfile ────────────────────────────────────────────────────
FROM python:3.11-slim

# Install OS-level dependencies needed by OpenCV (used by quality.py)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cache-friendly layer)
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy application source
COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY src/ ./src/

# Expose the FastAPI port
EXPOSE 8000

# DATABASE_URL is expected to be set via docker-compose env / volume.
# The default inside the container resolves to the shared /data volume.
ENV DATABASE_URL=sqlite:////data/product_categorization.db

WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
