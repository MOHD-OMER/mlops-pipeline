# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder — install all Python deps
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# System deps needed for scipy / scikit-learn compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --user \
        scikit-learn \
        numpy \
        pandas \
        fastapi \
        uvicorn[standard] \
        pydantic \
        mlflow \
        joblib \
        pyyaml \
        matplotlib \
        seaborn \
        evidently

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime — lean production image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY params.yaml      .
COPY src/             ./src/
COPY models/          ./models/

# Create dirs that might be needed at runtime
RUN mkdir -p reports mlruns

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the server
CMD ["python", "-m", "uvicorn", "src.serve:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
