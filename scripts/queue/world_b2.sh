#!/bin/bash
# Control for world_revise: the same library and plan, best of 2 with a
# blind second sample instead of the critique-and-rewrite. Starts when
# world_revise has finished (one 9B at a time).
cd "$(dirname "$0")/../.."
until grep -q DONE data/logs/world_revise.log; do sleep 30; done
FORGE_LIBRARY=release .venv/bin/python -u scripts/scorecard.py --world --n 30 --oneshot --api \
  --coder none --samples 2 --plan --tag world_b2 > data/logs/world_b2.log 2>&1
echo DONE >> data/logs/world_b2.log
