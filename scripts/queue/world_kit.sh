#!/bin/bash
# Phase 3 (2026-10-08): the 30 real-world requests with the repaired kit and
# the lesson plan -- against world_plan (same prompt, old kit).
cd "$(dirname "$0")/../.."
FORGE_LIBRARY=release .venv/bin/python -u scripts/scorecard.py --world --n 30 --oneshot --api \
  --coder none --samples 2 --plan --tag world_kit > data/logs/world_kit.log 2>&1
echo DONE >> data/logs/world_kit.log
