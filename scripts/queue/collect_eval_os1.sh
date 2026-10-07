#!/bin/bash
# Wait for nimbou/manim-forge-oneshot-eval (already pushed), download its
# scorecard dirs into data/scorecard/, judge them. The log is kept either way.
cd "$(dirname "$0")/../.."
K=.venv/bin/kaggle
log() { echo "[$(date -u +%H:%MZ)] $*"; }
sleep 300
until $K kernels status nimbou/manim-forge-oneshot-eval 2>&1 | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 300; done
log "eval: $($K kernels status nimbou/manim-forge-oneshot-eval 2>&1 | tail -1)"
rm -rf /tmp/os1_eval && mkdir -p /tmp/os1_eval
for i in 1 2 3; do $K kernels output nimbou/manim-forge-oneshot-eval -p /tmp/os1_eval > /dev/null 2>&1 && break; sleep 60; done
cp /tmp/os1_eval/*.log data/logs/kaggle_oneshot_eval.log 2>/dev/null
[ -d /tmp/os1_eval/scorecard ] || { log "no scorecards"; exit 1; }
cp -R /tmp/os1_eval/scorecard/* data/scorecard/ && log "copied: $(ls /tmp/os1_eval/scorecard | tr '\n' ' ')"
.venv/bin/python -u scripts/judge_sheets.py --model gemini-flash-lite-latest is_os1 held_os1 short_os1 is_os1_n3
.venv/bin/python -u scripts/judge_scenes.py --model gemini-flash-lite-latest is_os_base is_os1 is_os1_n3
log DONE
