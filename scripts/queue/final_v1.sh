#!/bin/bash
# v1.0's final numbers (2026-10-08): the shipped engine exactly -- Qwen3.5-9B
# on MLX, untuned, kit reference, best of 2 -- on the in-scope and held-out
# sets with the evaluation library (held-out topics removed). Graded by eye.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
for spec in "--inscope:is_v1" "--heldout:held_v1"; do
  IFS=: read set tag <<< "$spec"
  FORGE_LIBRARY=eval $PY -u scripts/scorecard.py $set --n 20 --oneshot --api --coder none \
    --samples 2 --tag $tag > data/logs/$tag.log 2>&1
done
echo DONE >> data/logs/held_v1.log
