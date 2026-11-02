# Start from a slim official Python 3.10 image (matches the project's target).
FROM python:3.10-slim

# All app files live under /app inside the container.
WORKDIR /app

# Install dependencies first, as their own layer: Docker caches this step and
# only re-runs it when requirements.txt changes, so code edits rebuild fast.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time, so the container starts fast
# and works offline (otherwise it fetches ~80MB on the first request). Cached as
# its own layer, so it only re-runs when requirements change.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the application code into the image.
COPY . .

# Document that the service listens on port 8000.
EXPOSE 8000

# Health check: Docker periodically pings /health so it knows the app is
# actually responding (not just that the process is alive). start-period gives
# the app time to boot and load the model before checks start counting.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the API. --host 0.0.0.0 makes it reachable from outside the container
# (localhost inside the container would not be).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
