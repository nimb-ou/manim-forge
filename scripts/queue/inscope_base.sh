#!/bin/bash
# In-scope baselines (2026-10-07): the shipped two-stage pipeline with planner
# v3 and v5, and the untuned one-shot, on forge/evaluate/inscope_prompts.json.
# One 7B at a time, in sequence; judged together with flash-lite at the end.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
$PY -u scripts/scorecard.py --inscope --n 20 --kit --relevance --planner adapters/mlx-planner3 \
  --coder adapters/mlx-coder8-kit --tag is_p3_k8 > data/logs/is_p3_k8.log 2>&1
$PY -u scripts/scorecard.py --inscope --n 20 --kit --relevance --planner adapters/mlx-planner5 \
  --coder adapters/mlx-coder8-kit --tag is_p5_k8 > data/logs/is_p5_k8.log 2>&1
$PY -u scripts/scorecard.py --inscope --n 20 --oneshot --api --coder none --tag is_os_base \
  > data/logs/is_os_base.log 2>&1
$PY -u scripts/judge_sheets.py --model gemini-flash-lite-latest is_p3_k8 is_p5_k8 is_os_base \
  > data/logs/judge_inscope_base.log 2>&1
echo DONE >> data/logs/judge_inscope_base.log
