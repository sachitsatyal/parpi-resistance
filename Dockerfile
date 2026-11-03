# syntax=docker/dockerfile:1.7
# One image for the CLI, batch inference and the web app (Hugging Face Space).
#   docker build -t parpi .
#   docker run --rm -p 7860:7860 -e PARPI_MODEL=hf://OWNER/parpi-resistance parpi

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder
# Same uv minor version that wrote uv.lock: older uv cannot read newer lockfiles.
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first (cached layer), resolved exactly from the committed lockfile.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra io --extra torch --extra app
COPY src ./src
# --no-editable installs the package into the venv, so the runtime needs no /app/src.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra io --extra torch --extra app

FROM python:${PYTHON_VERSION}-slim AS runtime
LABEL org.opencontainers.image.title="parpi-resistance" \
      org.opencontainers.image.description="Single-cell PARP-inhibitor resistance scoring" \
      org.opencontainers.image.licenses="MIT"
RUN useradd --create-home --uid 1000 user
# Keep the venv at the same path as in the builder: its scripts hard-code that path.
COPY --from=builder --chown=user /app/.venv /app/.venv
COPY --chown=user app /app/app
USER user
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib \
    GRADIO_ANALYTICS_ENABLED=False \
    PORT=7860
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ['PORT'] + '/')" || exit 1
WORKDIR /app/app
CMD ["python", "app.py"]
