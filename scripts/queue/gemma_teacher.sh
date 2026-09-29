#!/bin/zsh
# The Gemma kit teacher, following data/kit/gemma_plans.jsonl as gemma_arcs.py
# grows it. Rows are "kit-gemma": they count once the critic has judged them.
cd "$(dirname "$0")/../.." || exit 1
while true; do
  .venv/bin/python -u scripts/synth_kit_beats.py --provider gemma --model gemma-4-26b-a4b-it \
    --plans data/kit/gemma_plans.jsonl --limit 5000 --pause 1
  pgrep -f gemma_arcs.py >/dev/null || [ -n "$(grep -c . data/kit/gemma_plans.jsonl)" ] || break
  sleep 300
done
