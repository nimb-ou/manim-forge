#!/bin/bash
# v2.0 on the 50 dev requests (world 30 + the old fresh 20), to compare blind
# with world_v15 / fresh_v15. Waits for the solve trial; pauses on low battery.
cd "$(dirname "$0")/../.."
while pgrep -f "tag try_solve" >/dev/null; do sleep 30; done
wait_power() {
  while pmset -g batt | grep -q "Battery Power" && \
        [ "$(pmset -g batt | grep -o '[0-9]*%' | tr -d %)" -lt 30 ]; do sleep 120; done
}
A="--oneshot --api --coder none --samples 2 --revise 1 --plan --solve"
wait_power
FORGE_LIBRARY=release .venv/bin/python -u scripts/scorecard.py --world --n 30 $A --tag world_v20c > data/logs/world_v20c.log 2>&1
wait_power
FORGE_LIBRARY=release .venv/bin/python -u scripts/scorecard.py --fresh --n 20 $A --tag fresh_v20c > data/logs/fresh_v20c.log 2>&1
echo DONE > data/logs/dev_v20c.done
