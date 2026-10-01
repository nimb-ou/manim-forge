#!/bin/zsh
# judge_1001 ran out of gemini-flash-latest quota after 5 of 100 scenes
# (the free tier allows only a few dozen calls a day). The same five runs, judged
# fresh and all by gemini-flash-lite-latest, so the runs compared are judged alike.
cd "$(dirname "$0")/../.." || exit 1
for t in ab_held_names ab_held_scaf ab_short_names ab_short_scaf oracle_held; do
  rm -f data/scorecard/$t/judge.json
done
.venv/bin/python -u scripts/judge_sheets.py --model gemini-flash-lite-latest \
  ab_held_names ab_held_scaf ab_short_names ab_short_scaf oracle_held > data/logs/judge_1001b.log 2>&1
