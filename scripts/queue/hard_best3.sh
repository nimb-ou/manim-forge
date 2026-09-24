#!/bin/sh
# Kit mode on 12 hard titles (planner v3 + coder v2, kit via prompt only),
# after a one-title smoke. The raw baseline is hard_best (same planner and
# coder): 74 of 289 beats visual. Scored in data/scorecard/kit_v1.
cd "$(dirname "$0")/../.." || exit 1
nice -n 5 .venv/bin/python -u scripts/scorecard.py --n 1 --kit --max-beats 4 --tag smoke_kit > data/logs/scorecard_smoke.log 2>&1 || { echo "smoke failed"; exit 1; }
nice -n 5 .venv/bin/python -u scripts/scorecard.py --n 12 --kit --tag kit_v1 > data/logs/scorecard_kit_v1.log 2>&1
