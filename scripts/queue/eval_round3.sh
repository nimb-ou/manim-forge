#!/bin/sh
# After c5_fix: planner v4, then the relevance option (subject hint +
# resampling off-subject beats) on GRPO v2 and kit v5. Then self-training.
cd "$(dirname "$0")/../.." || exit 1
S="nice -n 5 .venv/bin/python -u scripts/scorecard.py --short --n 20 --max-beats 6 --kit"
$S --coder adapters/mlx-coder6-grpo --relevance --tag short_kit_grpo2_rel > data/logs/scorecard_short_kit_grpo2_rel.log 2>&1
$S --coder adapters/mlx-coder5-kit --relevance --tag short_kit_c5_rel > data/logs/scorecard_short_kit_c5_rel.log 2>&1
$S --coder adapters/mlx-coder5-kit --planner adapters/mlx-planner4 --tag short_kit_c5_p4 > data/logs/scorecard_short_kit_c5_p4.log 2>&1
exec scripts/queue/self_kit.sh
