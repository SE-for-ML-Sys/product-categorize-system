# ── monitoring/Dockerfile ─────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first (cache-friendly layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

EXPOSE 8001

# DB_PATH points to the shared named volume mounted by docker-compose
ENV DB_PATH=/data/product_categorization.db

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
