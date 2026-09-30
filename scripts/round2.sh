#!/bin/zsh
# Round 2 after the Oct 3 SFT, unattended:
#   1. wait for kit v8 (the supervisor's collect-kit8)
#   2. score it on the 20 held-out and 20 short prompts, cached planner-v3
#      plans, local vision judge -- against kit v6 with the same inference
#      (ab_held_scaf / ab_short_scaf)
#   3. if held-out is not worse (>= v6 - 3 points), round-2 self-training
#      with v8 on a Kaggle GPU (kaggle/selfgen), then collect: rows and
#      local-critic verdicts. The v9 SFT itself waits for a look.
cd "$(dirname "$0")/.." || exit 1
export PATH="$PWD/.venv/bin:$PATH"
log() { echo "$(date -u +%m-%dT%H:%MZ) $*"; }
until [ -f adapters/mlx-coder8-kit/adapters.safetensors ]; do sleep 600; done
log "kit v8 collected"
while pgrep -f "scorecard.py" >/dev/null; do sleep 60; done
pkill -f critic_loop.sh; pkill -f "critic_kit_scenes.py --local"
# v6 is re-scored with today's kit too: the kit keeps improving, and a
# baseline from an older kit would credit v8 with the kit's gains.
for c in 8 6; do
  for s in heldout short; do
    t=v${c}_${s%out}_r2
    python -u scripts/scorecard.py --$s --n 20 --kit --relevance --coder adapters/mlx-coder${c}-kit \
      --plans data/eval/plans_p3.json --tag $t > data/logs/scorecard_$t.log 2>&1
    python -u scripts/local_judge.py $t >> data/logs/scorecard_$t.log 2>&1
  done
done
share() { python -c "
import json,sys
t=sys.argv[1]; j=json.load(open(f'data/scorecard/{t}/judge_local.json'))
rows=json.load(open(f'data/scorecard/{t}/scorecard.json'))['rows']
print(sum(sum(v=='YES' for v in s['verdicts'].values()) for s in j.values())/sum(r['beats'] for r in rows))" $1; }
v8=$(share v8_held_r2); v6=$(share v6_held_r2); v8s=$(share v8_short_r2); v6s=$(share v6_short_r2)
log "held-out v8 $v8 vs v6 $v6; short v8 $v8s vs v6 $v6s" | tee data/kit/round2_decision
nohup scripts/queue/critic_loop.sh >> data/logs/kit_critic_local.log 2>&1 < /dev/null &
if python -c "import sys; sys.exit(0 if float('$v8') >= float('$v6') - 0.03 else 1)"; then
  log "v8 holds up: round-2 self-training on Kaggle" | tee -a data/kit/round2_decision
  python -u scripts/build_selfgen.py --limit 3000 || exit 1
  if kaggle datasets status nimbou/manim-forge-selfgen >/dev/null 2>&1; then
    kaggle datasets version -p kaggle/manim-forge-selfgen -m "round 2 arcs" -r zip -q
  else
    kaggle datasets create -p kaggle/manim-forge-selfgen -r zip -q
  fi
  sleep 300
  scripts/push_when_free.sh kaggle/selfgen || exit 1
  sleep 600
  while kaggle kernels status nimbou/manim-forge-selfgen 2>&1 | grep -qiE "running|queued"; do sleep 600; done
  log "selfgen: $(kaggle kernels status nimbou/manim-forge-selfgen 2>&1 | tail -1)"
  python -u scripts/collect_selfgen.py >> data/logs/collect_selfgen.log 2>&1
  log "collected: $(tail -1 data/logs/collect_selfgen.log)" | tee -a data/kit/round2_decision
else
  log "v8 worse than v6 on held-out: no round 2 until looked at" | tee -a data/kit/round2_decision
fi
