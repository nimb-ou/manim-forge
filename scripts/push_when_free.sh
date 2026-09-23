#!/bin/sh
# Push a kernel, retrying every 10 minutes while both GPU sessions are taken.
# Usage: scripts/push_when_free.sh coder|planner
cd "$(dirname "$0")/.." || exit 1
export PATH="$PWD/.venv/bin:$PATH"
i=0
while [ $i -lt 72 ]; do
  out=$(.venv/bin/python -u scripts/push_kernel.py "$1" 2>&1)
  echo "$(date -u +%H:%MZ) $out" | tail -2
  echo "$out" | grep -qi "error" || { echo "pushed $1"; exit 0; }
  i=$((i + 1))
  n=0; while [ $n -lt 40 ]; do sleep 15; n=$((n + 1)); done
done
echo "gave up pushing $1"; exit 1
