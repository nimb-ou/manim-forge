#!/bin/sh
# Train the kit coder locally once the teacher has written enough rows, then
# score it on the same 12 hard titles in kit mode.
cd "$(dirname "$0")/../.." || exit 1
nice -n 5 .venv/bin/python -u scripts/train_kit_coder.py --wait --min-rows 800 --epochs 2 > data/logs/train_kit_coder.log 2>&1 || exit 1
nice -n 5 .venv/bin/python -u scripts/scorecard.py --n 12 --kit --coder adapters/mlx-coder5-kit --tag kit_c5 > data/logs/scorecard_kit_c5.log 2>&1
