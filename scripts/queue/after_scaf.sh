#!/bin/zsh
# Restart the local critic loop once the held-out scorecard is done (the Mac
# swapped 5 GB with the 7B coder and the 4B critic loaded together).
cd "$(dirname "$0")/../.." || exit 1
while pgrep -f "scorecard.py" >/dev/null; do sleep 60; done
nohup scripts/queue/critic_loop.sh >> data/logs/kit_critic_local.log 2>&1 < /dev/null &
