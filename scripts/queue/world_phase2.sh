#!/bin/bash
# Phase 2 (2026-10-08): the same 30 real-world requests, same kit, with a
# lesson plan written first (--plan), then with the model's reasoning on
# (--think). Waits for the v1.0 baseline (world_base.sh).
cd "$(dirname "$0")/../.."
until grep -q DONE data/logs/world_v1.log 2>/dev/null; do sleep 60; done
for spec in "world_plan:--plan" "world_think:--think"; do
  IFS=: read tag flag <<< "$spec"
  FORGE_LIBRARY=release .venv/bin/python -u scripts/scorecard.py --world --n 30 --oneshot --api \
    --coder none --samples 2 $flag --tag $tag > data/logs/$tag.log 2>&1
done
echo DONE >> data/logs/world_think.log
