# ── Stage 1: build dependencies ──────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: production image ─────────────────────────────────
FROM python:3.11-slim

# Non-root user for security
RUN useradd -m appuser
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Create data directory (SQLite lives here)
RUN mkdir -p data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Serve API — uvicorn with 2 async workers
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
