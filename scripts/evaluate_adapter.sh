#!/usr/bin/env bash
# Phase 1's exit condition, end to end.
#
#   ./scripts/evaluate_adapter.sh                      # from Hugging Face
#   ./scripts/evaluate_adapter.sh path/to/peft-adapter # from disk
#
# Fetch the adapter, convert it to MLX and *prove* it loaded, run both evals
# against the recorded baselines, and print the verdict. About 90 minutes of
# local CPU: 100 benchmark trials and 81 hard-eval trials, each a generation
# plus a render.
#
# Every step refuses rather than degrades. An adapter that loads as a no-op
# would produce a complete, plausible, meaningless result -- the untuned
# numbers under the tuned name -- and that failure has already happened once
# in this project, silently, because mlx-lm loads adapters with strict=False.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="/Library/TeX/texbin:$PATH"
PY=./.venv/bin/python
REPO="${HF_ADAPTER_REPO:-nimitttt/manim-forge-sft}"
PEFT="${1:-}"
MLX="adapters/mlx-sft"

if [ -z "$PEFT" ]; then
    PEFT="adapters/kaggle-sft"
    echo "==> fetching $REPO"
    mkdir -p "$PEFT"
    $PY - "$REPO" "$PEFT" <<'EOF'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
print(f"    downloaded to {p}")
EOF
fi

echo
echo "==> converting to MLX and verifying it actually loads"
$PY scripts/peft_to_mlx.py --peft "$PEFT" --out "$MLX" --verify

echo
echo "==> single-scene benchmark (100 prompts, repair rounds=4, retrieval)"
$PY scripts/run_repair_benchmark.py --n 100 --rounds 4 --retrieval \
    --adapter "$MLX" --tag tuned100

echo
echo "==> hard eval (81 real 3Blue1Brown titles)"
$PY scripts/run_hard_eval.py --n 81 --backend local --rounds 4 --retrieval \
    --adapter "$MLX" --tag tuned81

echo
$PY scripts/compare_eval.py \
    --bench data/bench/tuned100_n100_r4.json \
    --hard  data/bench/tuned81_n81.json

echo
echo "Record these in docs/RESULTS.md with the mix hash from run.json."
