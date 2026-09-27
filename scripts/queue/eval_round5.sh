#!/bin/sh
# Kit v6 + relevance on the kit with the six new blocks; then self-training.
cd "$(dirname "$0")/../.." || exit 1
nice -n 5 .venv/bin/python -u scripts/scorecard.py --short --n 20 --max-beats 6 --kit \
  --coder adapters/mlx-coder6-kit --relevance --tag short_kit_c6_rel_k2 > data/logs/scorecard_short_kit_c6_rel_k2.log 2>&1
exec scripts/queue/self_kit.sh
