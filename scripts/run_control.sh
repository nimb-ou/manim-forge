#!/usr/bin/env bash
# The untuned control, with the byte-identical command the tuned run used.
#
# The 93% baseline on record lives in data/bench/rounds4_n100_r4.json, which
# is a bare list of trials: no n, no rounds, no retrieval flag, no model.
# The numbers match what the docs claim, and "the numbers match" is not the
# same as "the same experiment". Comparing a tuned run against it is a
# comparison with an unknown number of variables in it, and this project has
# already lost three results to exactly that.
#
# So: same flags, same prompts, same repair budget, same retrieval index,
# same machine, same day. The only difference is --adapter.
#
#   ./scripts/run_control.sh          # run it now
#   ./scripts/run_control.sh --after  # wait for a running eval first
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="/Library/TeX/texbin:$PATH"
PY=./.venv/bin/python

if [ "${1:-}" = "--after" ]; then
    echo "==> waiting for the tuned evaluation to finish"
    while pgrep -f "evaluate_adapter.sh" > /dev/null; do sleep 60; done
    echo "==> tuned evaluation finished at $(date -u '+%H:%M:%SZ')"
    sleep 10
fi

echo
echo "==> control: single-scene benchmark, no adapter (100 prompts, rounds=4, retrieval)"
$PY scripts/run_repair_benchmark.py --n 100 --rounds 4 --retrieval --tag ctrl100

echo
echo "==> control: hard eval, no adapter (81 real 3Blue1Brown titles)"
$PY scripts/run_hard_eval.py --n 81 --backend local --rounds 4 --retrieval --tag ctrl81

echo
echo "==> control finished at $(date -u '+%H:%M:%SZ')"
