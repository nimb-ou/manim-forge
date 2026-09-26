#!/bin/sh
# Collect GRPO v2, score it and kit v5 (with statement-level salvage), then
# go back to self-training.
cd "$(dirname "$0")/../.." || exit 1
.venv/bin/python -u scripts/collect_adapter.py --kernel nimbou/manim-forge-coder-grpo \
  --peft adapters/kaggle-coder6-grpo --mlx adapters/mlx-coder6-grpo > data/logs/collect_grpo6.log 2>&1
[ -f adapters/mlx-coder6-grpo/adapters.safetensors ] && \
  nice -n 5 .venv/bin/python -u scripts/scorecard.py --short --n 20 --max-beats 6 --kit \
    --coder adapters/mlx-coder6-grpo --tag short_kit_grpo2 > data/logs/scorecard_short_kit_grpo2.log 2>&1
nice -n 5 .venv/bin/python -u scripts/scorecard.py --short --n 20 --max-beats 6 --kit \
  --coder adapters/mlx-coder5-kit --tag short_kit_c5_salv > data/logs/scorecard_short_kit_c5_salv.log 2>&1
exec scripts/queue/self_kit.sh
