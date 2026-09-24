#!/bin/sh
# Kit vs raw Manim, same 12 hard titles, planner v3 + coder v2 -- after a
# one-title smoke of the kit path, which has not yet run against a model.
cd "$(dirname "$0")/../.." || exit 1
nice -n 5 .venv/bin/python -u scripts/scorecard.py --n 1 --kit --max-beats 4 --tag smoke_kit > data/logs/scorecard_smoke.log 2>&1 || { echo "smoke failed"; exit 1; }
nice -n 5 .venv/bin/python -u scripts/scorecard.py --n 12 --kit --tag kit_v1 > data/logs/scorecard_kit_v1.log 2>&1
nice -n 5 .venv/bin/python -u scripts/scorecard.py --n 12 --tag raw_v1 > data/logs/scorecard_raw_v1.log 2>&1
