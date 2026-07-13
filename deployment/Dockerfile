# Base image: Python 3.10 slim for a small serving footprint.
FROM python:3.10-slim

WORKDIR /app

# Install deps first for better Docker layer caching.
COPY deployment/requirements.txt /app/deployment/requirements.txt
RUN pip install --no-cache-dir -r /app/deployment/requirements.txt

# Copy application code (preserve src/, deployment/, configs/ layout for imports).
COPY src/ /app/src/
COPY deployment/ /app/deployment/
COPY configs/ /app/configs/

# Model weights are large — do NOT bake checkpoints into the image by default.
# Option (b) DEFAULT: mount at runtime, e.g.
#   docker run -p 8000:8000 -v $(pwd)/checkpoints:/app/checkpoints dbpedia-bert
# Option (a) ALTERNATIVE (fully self-contained image): uncomment the next line:
# COPY checkpoints/ /app/checkpoints/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
