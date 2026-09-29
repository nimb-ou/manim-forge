#!/bin/zsh
# Resume the local critic once the oracle diagnostic has finished (the Mac
# thrashed with both running: 3.4 GB swap, load 15).
cd "$(dirname "$0")/../.." || exit 1
while pgrep -f "scorecard.py.*oracle_held" >/dev/null; do sleep 60; done
nohup .venv/bin/python -u scripts/critic_kit_scenes.py --local --threshold=0.8 \
  >> data/logs/kit_critic_local.log 2>&1 < /dev/null &
