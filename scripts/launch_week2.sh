#!/bin/zsh
# At the Kaggle GPU reset (~2026-10-03 00:22Z): rebuild both datasets from
# everything on disk -- teacher, self-training and Claude-written rows,
# names-only subject hint -- and push kit coder v8 and planner v5 (two
# sessions at once). The supervisor's collect-kit8 / collect-planner5 wait on
# the *_pushed files. push_when_free.sh retries hourly if the quota is not
# back yet.
cd "$(dirname "$0")/.." || exit 1
export PATH="$PWD/.venv/bin:$PATH"
until [ "$(date -u +%Y%m%d%H%M)" -ge 202610030030 ]; do sleep 600; done
echo "$(date -u) building"
python -u scripts/claude_kit_scenes.py || exit 1
python -u scripts/build_kit_dataset.py || exit 1
python -u scripts/build_coder_dataset.py --which planner || exit 1
( python scripts/sync_kaggle_dataset.py kaggle/manim-forge-kit --ref nimbou/manim-forge-kit --expect train.jsonl \
  && scripts/push_when_free.sh kaggle/kit && sleep 90 && date > data/kit/kit8_pushed ) &
( python scripts/sync_kaggle_dataset.py kaggle/manim-forge-planner --ref nimbou/manim-forge-planner --expect train.jsonl \
  && scripts/push_when_free.sh kaggle/planner && sleep 90 && date > data/kit/planner5_pushed ) &
wait
echo "$(date -u) pushed"
