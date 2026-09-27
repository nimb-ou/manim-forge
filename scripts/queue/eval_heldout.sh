#!/bin/sh
# Held-out prompts: kit v6 + signature hints, with and without an exemplar.
cd "$(dirname "$0")/../.." || exit 1
S="nice -n 5 .venv/bin/python -u scripts/scorecard.py --heldout --n 20 --max-beats 6 --kit --relevance --coder adapters/mlx-coder6-kit"
$S --tag held_c6_sig > data/logs/scorecard_held_c6_sig.log 2>&1
$S --exemplar --tag held_c6_ex > data/logs/scorecard_held_c6_ex.log 2>&1
exec scripts/queue/self_kit.sh
