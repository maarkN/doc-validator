FROM python:3.12-slim

WORKDIR /app

# opencv-python-headless still links libglib (gthread) at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /usr/local/bin/uv

# Install locked dependencies (including the heavy ML extra) before copying
# the source so code changes do not invalidate the dependency layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra ml

COPY src ./src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "--factory", "app.main:create_app", "--host", "0.0.0.0", "--port", "8000"]
