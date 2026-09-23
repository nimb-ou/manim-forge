#!/bin/sh
# Overnight: evaluate each adapter as it lands, one MLX job at a time.
# Waits for any demo or other run_twostage/beat_eval to finish first -- two
# 7B models at once put this 16 GB Mac 6 GB into swap.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
idle() { while pgrep -f "scripts/(demo|run_twostage|beat_eval)[.]py" >/dev/null; do sleep 30; done; }
waitfor() { while [ ! -f "adapters/$1/adapters.safetensors" ]; do sleep 60; done; }
say() { echo "[$(date -u +%H:%MZ)] $*"; }

waitfor mlx-coder3; idle
say "coder v3 landed: beat eval"
nice -n 5 $PY -u scripts/beat_eval.py --adapter adapters/mlx-coder3 --tag coder3 > data/logs/beat_eval_coder3.log 2>&1
tail -7 data/logs/beat_eval_coder3.log
idle
say "two-stage planner2 + coder3"
nice -n 5 $PY -u scripts/run_twostage.py --n 8 --planner adapters/mlx-planner2 --coder adapters/mlx-coder3 --max-beats 12 --salvage --tag twostage_p2c3 > data/logs/twostage_p2c3.log 2>&1
grep "^assembled" data/logs/twostage_p2c3.log

waitfor mlx-planner3; idle
say "planner v3 landed: plan-only"
nice -n 5 $PY -u scripts/run_twostage.py --n 8 --hard --planner adapters/mlx-planner3 --max-beats 48 --plan-only --tag plans_v3_hard > data/logs/plans_v3_hard.log 2>&1
idle
say "two-stage planner3 + coder3"
nice -n 5 $PY -u scripts/run_twostage.py --n 8 --planner adapters/mlx-planner3 --coder adapters/mlx-coder3 --max-beats 24 --salvage --tag twostage_p3c3 > data/logs/twostage_p3c3.log 2>&1
grep "^assembled" data/logs/twostage_p3c3.log
say "chain done"
