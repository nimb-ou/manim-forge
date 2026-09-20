#!/usr/bin/env bash
# Sequential job queue for the heavy local work.
#
# One command, hours of work, live progress in Nimit's own terminal. Jobs run
# in order and the queue keeps going if one fails — a broken render should not
# cost the benchmark behind it.
#
#   ./scripts/queue.sh            run every job
#   ./scripts/queue.sh 2 3        run only jobs 2 and 3
#   ./scripts/queue.sh --list     show the queue and stop
#
# Every job is unbuffered and reads from /dev/null, so a tool that decides to
# ask an interactive question dies immediately instead of hanging silently.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/Library/TeX/texbin:$PATH"
export PYTHONUNBUFFERED=1
mkdir -p data/logs data/renders

M=./.venv/bin/manim
P=./.venv/bin/python

NAMES=(
  "render sphere_cube -qm    (verify the counting rewrite)"
  "render sphere_cube -qh    (1080p60 product quality)"
  "hard eval, rounds=4       (81 whole-explainer prompts)"
)
CMDS=(
  "$M render -qm --disable_caching --save_sections --media_dir data/renders forge/gold/sphere_cube.py SphereInCube"
  "$M render -qh --disable_caching --save_sections --media_dir data/renders forge/gold/sphere_cube.py SphereInCube"
  "$P -u scripts/run_hard_eval.py --n 81 --rounds 4 --retrieval --tag hard_r4"
)
LOGS=(
  "data/logs/render_sphere_qm.log"
  "data/logs/render_sphere_qh.log"
  "data/logs/hard_r4.log"
)

if [[ "${1:-}" == "--list" ]]; then
  for i in "${!NAMES[@]}"; do printf "  %d  %s\n" "$((i+1))" "${NAMES[$i]}"; done
  exit 0
fi

if [[ $# -gt 0 ]]; then SEL=("$@"); else SEL=($(seq 1 ${#NAMES[@]})); fi

QSTART=$(date +%s)
for n in "${SEL[@]}"; do
  i=$((n-1))
  [[ -z "${NAMES[$i]:-}" ]] && { echo "no job $n"; continue; }
  echo
  echo "================================================================"
  echo "  JOB $n/${#NAMES[@]}  ${NAMES[$i]}"
  echo "  started $(date '+%H:%M:%S')   log: ${LOGS[$i]}"
  echo "================================================================"
  S=$(date +%s)
  # tee, so it is watchable live AND kept on disk
  ${CMDS[$i]} < /dev/null 2>&1 | tee "${LOGS[$i]}"
  RC=${PIPESTATUS[0]}
  D=$(( $(date +%s) - S ))
  printf "  JOB %d %s in %dm %02ds\n" "$n" \
         "$([[ $RC -eq 0 ]] && echo OK || echo "FAILED rc=$RC")" $((D/60)) $((D%60))
done
echo
printf "queue finished in %dm %02ds\n" $(( ($(date +%s)-QSTART)/60 )) $(( ($(date +%s)-QSTART)%60 ))
find data/renders -name "SphereInCube.mp4" -not -path "*partial*" -exec ls -lh {} \; 2>/dev/null | awk '{print "  ", $5, $9}'
