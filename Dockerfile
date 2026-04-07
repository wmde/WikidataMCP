FROM python:3.13-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /uvx /bin/

# Setup the app in workspace
WORKDIR /workspace

# Install backend dependencies
COPY --chmod=755 pyproject.toml .
COPY --chmod=755 uv.lock .
RUN uv sync --locked

# Copy backend for production
COPY --chmod=755 . .

# Container start script
CMD ["uv", "run", "gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "--bind", "0.0.0.0:8000", "--timeout", "30", "--graceful-timeout", "15", "--max-requests", "1000", "--max-requests-jitter", "100", "--keep-alive", "5", "--access-logfile", "-", "--error-logfile", "-"]