#!/bin/zsh
# The local critic, re-run every 30 minutes so new teacher / Gemma / self
# scenes are judged before the SFT builds its dataset.
cd "$(dirname "$0")/../.." || exit 1
while true; do
  .venv/bin/python -u scripts/critic_kit_scenes.py --local --threshold=0.8
  sleep 1800
done
