#!/bin/zsh
# Headline judge after the Oct 1 quota reset, one pinned model for every run
# compared (the Sep 30 attempt ran out of quota part way).
cd "$(dirname "$0")/../.." || exit 1
until [ "$(date -u +%Y%m%d%H%M)" -ge 202610010715 ]; do sleep 300; done
for t in ab_held_names ab_held_scaf ab_short_names ab_short_scaf oracle_held; do
  rm -f data/scorecard/$t/judge.json
done
.venv/bin/python -u scripts/judge_sheets.py --model gemini-flash-latest \
  ab_held_names ab_held_scaf ab_short_names ab_short_scaf oracle_held > data/logs/judge_1001.log 2>&1
