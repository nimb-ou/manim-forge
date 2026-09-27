#!/bin/sh
# Kit v7 and kit v6 on the same kit (after the v7-notes fixes); GRPO v3
# when collected. Then self-training.
cd "$(dirname "$0")/../.." || exit 1
S="nice -n 5 .venv/bin/python -u scripts/scorecard.py --short --n 20 --max-beats 6 --kit --relevance"
$S --coder adapters/mlx-coder7-kit --tag short_kit_c7_k3 > data/logs/scorecard_short_kit_c7_k3.log 2>&1
$S --coder adapters/mlx-coder6-kit --tag short_kit_c6_k3 > data/logs/scorecard_short_kit_c6_k3.log 2>&1
while [ ! -f adapters/mlx-coder7-grpo/adapters.safetensors ]; do sleep 60; done
$S --coder adapters/mlx-coder7-grpo --tag short_kit_grpo3_k3 > data/logs/scorecard_short_kit_grpo3_k3.log 2>&1
exec scripts/queue/self_kit.sh
