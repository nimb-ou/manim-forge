#!/bin/sh
cd "$(dirname "$0")/../.." || exit 1
exec nice -n 5 .venv/bin/python -u scripts/run_twostage.py --n 24 --hard --planner adapters/mlx-planner3 --coder adapters/mlx-coder2 --max-beats 24 --beat-tokens 1400 --salvage --tag hard_best3
