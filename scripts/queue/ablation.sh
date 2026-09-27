#!/bin/sh
# Kit v6 on identical plans (cache data/eval/plans_p3.json), 40 prompts
# (20 short + 20 held-out), one ingredient at a time. Then self-training.
cd "$(dirname "$0")/../.." || exit 1
P="--plans data/eval/plans_p3.json --max-beats 6 --kit --coder adapters/mlx-coder6-kit"
S="nice -n 5 .venv/bin/python -u scripts/scorecard.py"
for cfg in "sig:--relevance" "plain:" "ex:--relevance --exemplar" "names:--relevance --no-signatures"; do
  tag=${cfg%%:*}; flags=${cfg#*:}
  $S --short --n 20 $P $flags --tag ab_short_$tag > data/logs/scorecard_ab_short_$tag.log 2>&1
  $S --heldout --n 20 $P $flags --tag ab_held_$tag > data/logs/scorecard_ab_held_$tag.log 2>&1
done
exec scripts/queue/self_kit.sh
