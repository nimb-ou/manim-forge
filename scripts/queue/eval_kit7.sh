#!/bin/sh
# Kit v7 (hand-written rows for the new blocks, hints in prompts) with
# relevance, on the current kit; then self-training resumes.
cd "$(dirname "$0")/../.." || exit 1
nice -n 5 .venv/bin/python -u scripts/scorecard.py --short --n 20 --max-beats 6 --kit \
  --coder adapters/mlx-coder7-kit --relevance --tag short_kit_c7_rel > data/logs/scorecard_short_kit_c7_rel.log 2>&1
exec scripts/queue/self_kit.sh
