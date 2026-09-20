#!/usr/bin/env bash
# Stop cleanly, then start the worker pool. Safe to run repeatedly.
set -uo pipefail
cd "$(dirname "$0")/.."
./scripts/forge_stop.sh
export PATH="/Library/TeX/texbin:$PATH"
nohup ./.venv/bin/python -u scripts/forge_run.py --interval 60 \
      > data/logs/orchestrator.log 2>&1 &
echo "forge_run started (pid $!)  ->  data/logs/orchestrator.log"
