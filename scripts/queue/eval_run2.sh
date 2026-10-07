#!/bin/bash
# 2026-10-07: fetch run 1's scorecards via the packer kernel (one tar), then
# push eval run 2 (base 7B vs Qwen3.5-9B, untuned, kit API) and fetch its tar.
cd "$(dirname "$0")/../.."
# Every Kaggle call gets a deadline: on 2026-10-07 one `kernels status`
# hung for 2.5 h and the whole queue with it. macOS has no timeout(1).
kt() { local t=$1; shift; perl -e 'alarm shift; exec @ARGV' "$t" .venv/bin/kaggle "$@"; }
K="kt 180"
log() { echo "[$(date -u +%H:%MZ)] $*"; }
fetch() {  # kernel, dir
  rm -rf "$2"; mkdir -p "$2"
  for i in 1 2 3 4 5; do
    kt 1800 kernels output "$1" -p "$2" --file-pattern 'scorecards\.tar\.gz$' -q > /dev/null 2>&1
    [ -s "$2/scorecards.tar.gz" ] && break; sleep 120
  done
  tar -xzf "$2/scorecards.tar.gz" -C data/scorecard && log "unpacked $(tar -tzf "$2/scorecards.tar.gz" | cut -d/ -f1 | sort -u | tr '\n' ' ')"
}
until $K kernels status nimbou/manim-forge-packer 2>&1 | grep -qE "COMPLETE|ERROR"; do sleep 60; done
log "packer: $($K kernels status nimbou/manim-forge-packer 2>&1 | tail -1)"
fetch nimbou/manim-forge-packer /tmp/run1_tar
kt 600 kernels push -p kaggle/oneshot_eval 2>&1 | tail -1
sleep 300
until $K kernels status nimbou/manim-forge-oneshot-eval 2>&1 | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 300; done
log "eval run 2: $($K kernels status nimbou/manim-forge-oneshot-eval 2>&1 | tail -1)"
$K kernels output nimbou/manim-forge-oneshot-eval -p /tmp/run2_log --file-pattern '.*\.log$' -q > /dev/null 2>&1
cp /tmp/run2_log/*.log data/logs/kaggle_eval_run2.log 2>/dev/null
fetch nimbou/manim-forge-oneshot-eval /tmp/run2_tar
log DONE
