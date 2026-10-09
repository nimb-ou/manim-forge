#!/bin/bash
# v1.5 phase 3b: the plan run with one critique-and-rewrite attempt in place
# of the blind second sample (forge/app/critique.py).
cd "$(dirname "$0")/../.."
FORGE_LIBRARY=release .venv/bin/python -u scripts/scorecard.py --world --n 30 --oneshot --api \
  --coder none --samples 2 --revise 1 --plan --tag world_revise > data/logs/world_revise.log 2>&1
echo DONE >> data/logs/world_revise.log
