#!/bin/sh
# Push a kernel, retrying while Kaggle refuses: every 10 minutes while both
# GPU sessions are taken, hourly once the weekly 30-hour quota is spent
# (it resets weekly; up to 4 days of waiting).
# Usage: scripts/push_when_free.sh coder|planner
cd "$(dirname "$0")/.." || exit 1
export PATH="$PWD/.venv/bin:$PATH"
end=$(( $(date +%s) + 4 * 86400 ))
while [ "$(date +%s)" -lt "$end" ]; do
  out=$(.venv/bin/python -u scripts/push_kernel.py "$1" 2>&1)
  echo "$(date -u +%m-%dT%H:%MZ) $(echo "$out" | tail -1)"
  echo "$out" | grep -qi "error" || { echo "pushed $1"; exit 0; }
  steps=40
  echo "$out" | grep -qi "quota" && steps=240
  n=0; while [ $n -lt $steps ]; do sleep 15; n=$((n + 1)); done
done
echo "gave up pushing $1"; exit 1
