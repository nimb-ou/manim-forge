#!/bin/sh
# One MLX job at a time. pgrep patterns use [.] so this script's own command
# line never matches; two 7B models at once put this Mac 6 GB into swap.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
idle() { while pgrep -f "scripts/(demo|run_twostage|beat_eval)[.]py" >/dev/null; do sleep 30; done; }
say() { echo "[$(date -u +%H:%MZ)] $*"; }

idle; say "planner v3 plan-only, sampled, 8 hard titles"
nice -n 5 $PY -u scripts/run_twostage.py --n 8 --hard --planner adapters/mlx-planner3 --max-beats 48 --plan-only --tag plans_v3_sampled > data/logs/plans_v3_sampled.log 2>&1
grep -E "beats/arc|declared" data/logs/plans_v3_sampled.log

# The better planner by beats-before-first-repeat goes to the hard eval.
v2=$(grep "beats/arc" data/logs/plans_v2_sampled.log | awk '{print $2}')
v3=$(grep "beats/arc" data/logs/plans_v3_sampled.log | awk '{print $2}')
P=adapters/mlx-planner2
[ "$(echo "$v3 > $v2" | bc)" = "1" ] && P=adapters/mlx-planner3
say "planner v2 $v2 vs v3 $v3 beats/arc -> hard eval with $P"

idle; say "hard eval, 24 titles, all harness fixes"
nice -n 5 $PY -u scripts/run_twostage.py --n 24 --hard --planner $P --coder adapters/mlx-coder2 --max-beats 24 --salvage --tag hard_best > data/logs/hard_best.log 2>&1
tail -6 data/logs/hard_best.log
say "chain done"
