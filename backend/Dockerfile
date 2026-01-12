# =====================================================
# Floodingnaque Backend - Production Dockerfile
# Multi-stage build for optimized production image
# =====================================================
# Build: docker build -t floodingnaque-api:latest .
# Run:   docker run -p 5000:5000 --env-file .env floodingnaque-api:latest
# =====================================================

# ===================
# Build Arguments
# ===================
ARG PYTHON_VERSION=3.12

# =====================================================
# Stage 1: Builder - Install dependencies
# =====================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
WORKDIR /build
COPY requirements.txt .

# Install dependencies with optimizations
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# =====================================================
# Stage 2: Production - Minimal runtime image
# =====================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS production

# Labels for container metadata
LABEL maintainer="Floodingnaque Team" \
      version="2.0.0" \
      description="Floodingnaque Flood Prediction API" \
      org.opencontainers.image.source="https://github.com/floodingnaque/floodingnaque"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    # Application settings
    APP_ENV=production \
    PORT=5000 \
    HOST=0.0.0.0 \
    PYTHONPATH=/app \
    # Gunicorn settings
    GUNICORN_WORKERS=4 \
    GUNICORN_THREADS=2 \
    GUNICORN_TIMEOUT=120 \
    GUNICORN_WORKER_CLASS=sync \
    GUNICORN_MAX_REQUESTS=1000 \
    GUNICORN_MAX_REQUESTS_JITTER=50

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/*

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Create application directories
RUN mkdir -p /app/models /app/data /app/logs /app/uploads /app/backups /app/.cache /app/certs && \
    chown -R appuser:appgroup /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup main.py ./
COPY --chown=appuser:appgroup alembic.ini ./
COPY --chown=appuser:appgroup alembic/ ./alembic/

# Copy model files (JSON metadata and joblib models)
COPY --chown=appuser:appgroup models/*.json ./models/
COPY --chown=appuser:appgroup models/*.joblib ./models/

# Copy SSL certificates for database connection (Supabase)
COPY --chown=appuser:appgroup prod-ca-2021.crt ./certs/

# Copy data files for reference
COPY --chown=appuser:appgroup data/*.csv ./data/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command: Run with Gunicorn (Datadog APM tracing if available)
CMD ["sh", "-c", "\
if command -v ddtrace-run >/dev/null 2>&1; then \
  exec ddtrace-run gunicorn main:application \
    --bind ${HOST}:${PORT} \
    --workers ${GUNICORN_WORKERS} \
    --threads ${GUNICORN_THREADS} \
    --timeout ${GUNICORN_TIMEOUT} \
    --worker-class ${GUNICORN_WORKER_CLASS} \
    --max-requests ${GUNICORN_MAX_REQUESTS} \
    --max-requests-jitter ${GUNICORN_MAX_REQUESTS_JITTER} \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --enable-stdio-inheritance \
    --preload; \
else \
  exec gunicorn main:application \
    --bind ${HOST}:${PORT} \
    --workers ${GUNICORN_WORKERS} \
    --threads ${GUNICORN_THREADS} \
    --timeout ${GUNICORN_TIMEOUT} \
    --worker-class ${GUNICORN_WORKER_CLASS} \
    --max-requests ${GUNICORN_MAX_REQUESTS} \
    --max-requests-jitter ${GUNICORN_MAX_REQUESTS_JITTER} \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --enable-stdio-inheritance \
    --preload; \
fi"]

# =====================================================
# Stage 3: Development - Full dev environment
# =====================================================
FROM production AS development

# Switch back to root to install dev dependencies
USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    git \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development dependencies
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt || true

# Set development environment
ENV APP_ENV=development \
    FLASK_DEBUG=True \
    LOG_LEVEL=DEBUG

# Switch back to non-root user
USER appuser

# Override command for development (with hot reload via Flask)
CMD ["python", "main.py"]
