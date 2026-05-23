#!/bin/bash
# start.sh — Docker entrypoint
# Starts the daily scheduler (background) then Streamlit (foreground).
# Scheduler fires at 04:00 UTC = 09:30 IST on Mon–Fri.

set -euo pipefail

echo "[start.sh] Creating persistent directories..."
mkdir -p .cache experiments data_store figures

echo "[start.sh] Starting daily agent scheduler (04:00 UTC / 09:30 IST, Mon–Fri)..."
python scripts/run_scheduler.py &
SCHEDULER_PID=$!
echo "[start.sh] Scheduler PID: $SCHEDULER_PID"

echo "[start.sh] Starting Streamlit dashboard on port ${PORT:-8501}..."
exec python -m streamlit run src/app/streamlit_app.py \
    --server.port "${PORT:-8501}" \
    --server.address "0.0.0.0" \
    --server.headless "true" \
    --browser.gatherUsageStats "false"
