#!/bin/bash
# One-shot v1 (2026-10-07): wait for the in-scope baselines, collect the
# adapter from Kaggle, then score it on in-scope (1 and 3 samples), held-out
# and short, and judge everything in-scope with both judges.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
until grep -q DONE data/logs/judge_inscope_base.log 2>/dev/null; do sleep 60; done
$PY -u scripts/collect_adapter.py --kernel nimbou/manim-forge-oneshot-sft \
  --peft adapters/kaggle-oneshot --mlx adapters/mlx-oneshot --max-hours 12 \
  > data/logs/collect_oneshot.log 2>&1
[ -f adapters/mlx-oneshot/adapters.safetensors ] || { echo "no adapter" >> data/logs/collect_oneshot.log; exit 1; }
for spec in "--inscope:is_os1:1" "--inscope:is_os1_n3:3" "--heldout:held_os1:1" "--short:short_os1:1"; do
  IFS=: read set tag n <<< "$spec"
  $PY -u scripts/scorecard.py $set --n 20 --oneshot --coder adapters/mlx-oneshot \
    --samples $n --tag $tag > data/logs/$tag.log 2>&1
done
$PY -u scripts/judge_sheets.py --model gemini-flash-lite-latest is_os1 is_os1_n3 held_os1 short_os1 \
  > data/logs/judge_os1.log 2>&1
$PY -u scripts/judge_scenes.py --model gemini-flash-lite-latest is_p3_k8 is_p5_k8 is_os_base is_os1 is_os1_n3 \
  >> data/logs/judge_os1.log 2>&1
echo DONE >> data/logs/judge_os1.log
