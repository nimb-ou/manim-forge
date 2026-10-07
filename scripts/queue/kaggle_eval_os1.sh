#!/bin/bash
# One-shot v1 scored on Kaggle (2026-10-07; the Mac is on battery): wait for
# the SFT kernel, push kaggle/oneshot_eval (which mounts its output), wait,
# download the scorecard dirs into data/scorecard/, then judge them.
cd "$(dirname "$0")/../.."
K=.venv/bin/kaggle
log() { echo "[$(date -u +%H:%MZ)] $*"; }
until $K kernels status nimbou/manim-forge-oneshot-sft 2>&1 | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 600; done
log "sft: $($K kernels status nimbou/manim-forge-oneshot-sft 2>&1 | tail -1)"
$K kernels push -p kaggle/oneshot_eval 2>&1 | tail -1
sleep 120
until $K kernels status nimbou/manim-forge-oneshot-eval 2>&1 | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 600; done
log "eval: $($K kernels status nimbou/manim-forge-oneshot-eval 2>&1 | tail -1)"
rm -rf /tmp/os1_eval && mkdir -p /tmp/os1_eval
for i in 1 2 3; do $K kernels output nimbou/manim-forge-oneshot-eval -p /tmp/os1_eval > /dev/null 2>&1 && break; sleep 60; done
cp -R /tmp/os1_eval/scorecard/* data/scorecard/ && log "copied: $(ls /tmp/os1_eval/scorecard)"
grep -E "done in|FAILED|all done" /tmp/os1_eval/*.log 2>/dev/null | tail -8
.venv/bin/python -u scripts/judge_sheets.py --model gemini-flash-lite-latest is_os1 held_os1 short_os1 is_os1_n3
.venv/bin/python -u scripts/judge_scenes.py --model gemini-flash-lite-latest is_p3_k8 is_p5_k8 is_os_base is_os1 is_os1_n3
log DONE
