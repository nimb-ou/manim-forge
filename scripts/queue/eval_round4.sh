#!/bin/sh
# Kit v6 (teacher + self rows) with the relevance option, then planner v4.
cd "$(dirname "$0")/../.." || exit 1
S="nice -n 5 .venv/bin/python -u scripts/scorecard.py --short --n 20 --max-beats 6 --kit"
$S --coder adapters/mlx-coder6-kit --relevance --tag short_kit_c6_rel > data/logs/scorecard_short_kit_c6_rel.log 2>&1
$S --coder adapters/mlx-coder6-grpo --relevance --planner adapters/mlx-planner4 --tag short_kit_grpo2_rel_p4 > data/logs/scorecard_short_kit_grpo2_rel_p4.log 2>&1
exec scripts/queue/self_kit.sh
