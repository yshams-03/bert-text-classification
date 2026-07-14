# Multi-stage build: builder stage for deps, minimal runtime stage
FROM python:3.10-slim as builder

WORKDIR /build

# Install system build tools for wheels (e.g., torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels
COPY deployment/requirements.txt .
RUN pip install --user --no-cache-dir \
    --only-binary :all: \
    --no-warn-script-location \
    -r requirements.txt


# Runtime stage: minimal Python image with only runtime deps
FROM python:3.10-slim

WORKDIR /app

# Copy pre-built wheels from builder
COPY --from=builder /root/.local /root/.local

# Set PATH to use local pip packages
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy application code (preserve src/, deployment/, configs/ layout for imports)
COPY src/ /app/src/
COPY deployment/ /app/deployment/
COPY configs/ /app/configs/

# Model checkpoints are large — mount at runtime as a volume (do not bake into image)
# For self-contained image: uncomment next line and rebuild
# COPY checkpoints/ /app/checkpoints/

EXPOSE 8000

# Health check: verify FastAPI is responding on /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
