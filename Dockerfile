# Linux base image with Python 3.11 — matches the training environment.
# "slim" excludes build tools and docs, cutting the image size substantially.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies BEFORE copying source code.
# Docker caches each layer; putting requirements first means code edits
# don't trigger a full reinstall on every rebuild.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only what the API needs at runtime
COPY src/app.py src/
COPY models/model.pkl models/

# Document the port. This does not publish it — that happens at `docker run`.
EXPOSE 8000

# 0.0.0.0 binds all interfaces. Using 127.0.0.1 here would make the API
# unreachable from outside the container.
CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}"]