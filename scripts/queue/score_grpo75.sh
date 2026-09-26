#!/bin/sh
cd "$(dirname "$0")/../.." || exit 1
nice -n 5 .venv/bin/python -u scripts/scorecard.py --short --n 20 --max-beats 6 --kit --coder adapters/mlx-coder6-grpo75 --tag short_kit_grpo75 > data/logs/scorecard_short_kit_grpo75.log 2>&1
