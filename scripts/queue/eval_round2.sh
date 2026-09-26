#!/bin/sh
# Same kit (after the fuzz fixes) for all three: GRPO v2, kit v5, and kit v5
# under planner v4. Then back to self-training.
cd "$(dirname "$0")/../.." || exit 1
S="nice -n 5 .venv/bin/python -u scripts/scorecard.py --short --n 20 --max-beats 6 --kit"
$S --coder adapters/mlx-coder6-grpo --tag short_kit_grpo2 > data/logs/scorecard_short_kit_grpo2.log 2>&1
$S --coder adapters/mlx-coder5-kit --tag short_kit_c5_fix > data/logs/scorecard_short_kit_c5_fix.log 2>&1
$S --coder adapters/mlx-coder5-kit --planner adapters/mlx-planner4 --tag short_kit_c5_p4 > data/logs/scorecard_short_kit_c5_p4.log 2>&1
exec scripts/queue/self_kit.sh
