#!/bin/zsh
# Kit v6 on the held-out plans with today's kit (stacked equations, the
# region rule, label fixes): did the kit work since kit930 move it? The
# local critic pauses so the 7B coder and the 4B judge don't share memory
# with it.
cd "$(dirname "$0")/../.." || exit 1
export PATH="$PWD/.venv/bin:$PATH"
pkill -f critic_loop.sh; pkill -f "critic_kit_scenes.py --local"
t=ab_held_kit1001
python -u scripts/scorecard.py --heldout --n 20 --kit --relevance --coder adapters/mlx-coder6-kit \
  --plans data/eval/plans_p3.json --tag $t > data/logs/scorecard_$t.log 2>&1
python -u scripts/local_judge.py $t >> data/logs/scorecard_$t.log 2>&1
nohup scripts/queue/critic_loop.sh >> data/logs/kit_critic_local.log 2>&1 < /dev/null &
echo "done $(date -u)" >> data/logs/scorecard_$t.log
