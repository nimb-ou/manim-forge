#!/bin/bash
# The fresh set with the self-trained adapter, after final_v15.
cd "$(dirname "$0")/../.."
until [ -f data/logs/final_v15.done ]; do sleep 30; done
FORGE_LIBRARY=release .venv/bin/python -u scripts/scorecard.py --fresh --n 20 --oneshot --api --samples 2 \
  --revise 1 --plan --coder adapters/mlx-selftrain --base mlx-community/Qwen3.5-9B-MLX-4bit \
  --tag fresh_v15t > data/logs/fresh_v15t.log 2>&1
echo DONE >> data/logs/fresh_v15t.log
