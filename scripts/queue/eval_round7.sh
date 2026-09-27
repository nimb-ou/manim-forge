#!/bin/sh
# Hints with signatures (how to call each block) and shading kept on the
# axes: kit v6 and GRPO v3. Then self-training.
cd "$(dirname "$0")/../.." || exit 1
S="nice -n 5 .venv/bin/python -u scripts/scorecard.py --short --n 20 --max-beats 6 --kit --relevance"
$S --coder adapters/mlx-coder6-kit --tag short_kit_c6_sig > data/logs/scorecard_short_kit_c6_sig.log 2>&1
$S --coder adapters/mlx-coder7-grpo --tag short_kit_grpo3_sig > data/logs/scorecard_short_kit_grpo3_sig.log 2>&1
exec scripts/queue/self_kit.sh
