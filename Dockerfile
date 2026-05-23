FROM python:3.10-slim

LABEL maintainer="AgentQuant"
LABEL description="AgentQuant — AI Paper Trading Dashboard + Daily Scheduled Agent"

WORKDIR /app

# ── System deps (scipy / numpy native extensions) ─────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps (layer-cached until pyproject.toml changes) ───────────────────
COPY pyproject.toml ./

# Install all groups needed in production: core + llm + regime
# 'data' group (fredapi, alpaca) is optional; skip to keep image lean
RUN pip install --no-cache-dir -e ".[llm,regime]"

# ── Copy source ────────────────────────────────────────────────────────────────
COPY . .

# ── Persistent data directories (overridden by mounted volumes in prod) ────────
RUN mkdir -p .cache experiments data_store figures

# ── Entrypoint script ──────────────────────────────────────────────────────────
COPY start.sh /start.sh
RUN chmod +x /start.sh

# ── Streamlit port (Railway injects PORT at runtime) ──────────────────────────
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

ENTRYPOINT ["/start.sh"]
