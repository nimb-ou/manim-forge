#!/bin/sh
# Kit v6 + names hint on the cached plans, kit after the teacher-reported
# fixes; judged by the vision judge. Then self-training.
cd "$(dirname "$0")/../.." || exit 1
P="--plans data/eval/plans_p3.json --max-beats 6 --kit --relevance --coder adapters/mlx-coder6-kit"
S="nice -n 5 .venv/bin/python -u scripts/scorecard.py"
$S --short --n 20 $P --tag kf_short_names > data/logs/scorecard_kf_short_names.log 2>&1
$S --heldout --n 20 $P --tag kf_held_names > data/logs/scorecard_kf_held_names.log 2>&1
.venv/bin/python -u scripts/judge_sheets.py kf_short_names kf_held_names > data/logs/judge_kf.log 2>&1
exec scripts/queue/self_kit.sh
