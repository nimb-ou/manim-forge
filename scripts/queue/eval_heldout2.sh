#!/bin/sh
# Held-out, seeded plans: kit v6 + hints, without and with an exemplar.
cd "$(dirname "$0")/../.." || exit 1
S="nice -n 5 .venv/bin/python -u scripts/scorecard.py --heldout --n 20 --max-beats 6 --kit --relevance --coder adapters/mlx-coder6-kit"
$S --tag held_c6_sig_s > data/logs/scorecard_held_c6_sig_s.log 2>&1
$S --exemplar --tag held_c6_ex_s > data/logs/scorecard_held_c6_ex_s.log 2>&1
exec scripts/queue/self_kit.sh
