#!/bin/sh
# Rejection-sampling data from the kit coder itself, until stopped. Resumable.
cd "$(dirname "$0")/../.." || exit 1
exec nice -n 5 .venv/bin/python -u scripts/self_kit_beats.py --coder adapters/mlx-coder5-kit >> data/logs/self_kit.log 2>&1
