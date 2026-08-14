# MorganVuoksi Terminal - Railway.app Optimized Dockerfile
# Multi-stage build for optimal size and performance

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

# Set build arguments for optimization
ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_NO_CACHE_DIR=1
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements with retry logic
COPY requirements-railway.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt || \
    (echo "Retrying with individual core packages..." && \
     pip install --no-cache-dir streamlit pandas numpy plotly yfinance scikit-learn && \
     pip install --no-cache-dir -r /tmp/requirements.txt)

# Stage 2: Runtime
FROM python:3.11-slim as runtime

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Environment variables for Railway
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    PYTHONPATH=/app \
    PORT=8501

# Create non-root user for security
RUN useradd -m -u 1000 streamlit && \
    mkdir -p /app && \
    chown -R streamlit:streamlit /app

# Set work directory
WORKDIR /app

# Copy application code
COPY --chown=streamlit:streamlit . .

# Create necessary directories with proper permissions
RUN mkdir -p logs outputs models/saved_models data && \
    chown -R streamlit:streamlit logs outputs models data

# Make scripts executable
RUN chmod +x startup.py

# Switch to non-root user
USER streamlit

# Expose port (Railway will override with $PORT)
EXPOSE 8501

# Health check optimized for Railway
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Optimized startup command
CMD ["python", "startup.py"] 