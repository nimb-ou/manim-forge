#!/bin/sh
# Kit vs raw Manim, same 12 hard titles, planner v3 + coder v2.
cd "$(dirname "$0")/../.." || exit 1
nice -n 5 .venv/bin/python -u scripts/scorecard.py --n 12 --kit --tag kit_v1 > data/logs/scorecard_kit_v1.log 2>&1
nice -n 5 .venv/bin/python -u scripts/scorecard.py --n 12 --tag raw_v1 > data/logs/scorecard_raw_v1.log 2>&1
