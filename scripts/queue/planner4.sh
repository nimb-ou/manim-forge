#!/bin/sh
# Planner v4 vs v3 (ab_*_names), coder kit v6 + names hint: plans from
# planner v4 cached separately, same seeds. Then self-training.
cd "$(dirname "$0")/../.." || exit 1
P="--plans data/eval/plans_p4.json --planner adapters/mlx-planner4 --max-beats 6 --kit --relevance --coder adapters/mlx-coder6-kit"
S="nice -n 5 .venv/bin/python -u scripts/scorecard.py"
$S --short --n 20 $P --tag ab_short_names_p4 > data/logs/scorecard_ab_short_names_p4.log 2>&1
$S --heldout --n 20 $P --tag ab_held_names_p4 > data/logs/scorecard_ab_held_names_p4.log 2>&1
exec scripts/queue/self_kit.sh
