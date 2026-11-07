# --- UI -------------------------------------------------------------------------------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# --- backend --------------------------------------------------------------------------------------
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY doccheck/ doccheck/
COPY scripts/ scripts/
COPY data/ data/
COPY runs/demo/ runs/demo/
RUN uv sync --frozen --no-dev
COPY --from=web /web/dist web/dist
EXPOSE 8000
# Replay of recorded runs works without an LLM key; the embedding model is downloaded on the first live check.
CMD ["uv", "run", "--no-dev", "uvicorn", "doccheck.api:app", "--host", "0.0.0.0", "--port", "8000"]
