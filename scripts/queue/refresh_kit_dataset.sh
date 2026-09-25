#!/bin/sh
# Until the kit coder is pushed: every hour, rebuild the kit dataset from the
# teacher's growing output and publish a new version, so the Kaggle run
# trains on everything written by then.
cd "$(dirname "$0")/../.." || exit 1
export PATH="$PWD/.venv/bin:$PATH"
while [ ! -f data/kit/kit5_pushed ]; do
  .venv/bin/python scripts/build_kit_dataset.py 2>&1 | tail -1
  .venv/bin/python -u scripts/sync_kaggle_dataset.py kaggle/manim-forge-kit --ref nimbou/manim-forge-kit --expect train.jsonl valid.jsonl 2>&1 | tail -1
  n=0; while [ $n -lt 240 ] && [ ! -f data/kit/kit5_pushed ]; do sleep 15; n=$((n+1)); done
done
