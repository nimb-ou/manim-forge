#!/bin/bash
# Phase 1 (2026-10-08): v1.0 as shipped on the 30 real-world requests, with
# the release library like the app. The baseline every v1.1+ change must beat.
cd "$(dirname "$0")/../.."
FORGE_LIBRARY=release .venv/bin/python -u scripts/scorecard.py --world --n 30 --oneshot --api \
  --coder none --samples 2 --tag world_v1 > data/logs/world_v1.log 2>&1
echo DONE >> data/logs/world_v1.log
