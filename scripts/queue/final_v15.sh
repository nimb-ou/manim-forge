#!/bin/bash
# v1.5 finish, after the v1.0 fresh run (one 9B at a time):
#  1. the self-trained adapter loads and changes the model
#  2. dev set, adapter       (world_tuned)
#  3. dev set, untuned       (world_v15)  -- same engine and kit: the control
#  4. fresh set, v1.5 untuned (fresh_v15) -- the final test against fresh_v10
cd "$(dirname "$0")/../.."
until grep -q DONE data/logs/fresh_v10.log; do sleep 30; done
.venv/bin/python scripts/peft_to_mlx.py --peft adapters/kaggle-selftrain --out adapters/mlx-selftrain \
  --base mlx-community/Qwen3.5-9B-MLX-4bit --verify > data/logs/verify_selftrain.log 2>&1 || exit 1
W="--world --n 30 --oneshot --api --samples 2 --revise 1 --plan"
FORGE_LIBRARY=release .venv/bin/python -u scripts/scorecard.py $W --coder adapters/mlx-selftrain \
  --base mlx-community/Qwen3.5-9B-MLX-4bit --tag world_tuned > data/logs/world_tuned.log 2>&1
FORGE_LIBRARY=release .venv/bin/python -u scripts/scorecard.py $W --coder none --tag world_v15 > data/logs/world_v15.log 2>&1
FORGE_LIBRARY=release .venv/bin/python -u scripts/scorecard.py --fresh --n 20 --oneshot --api --samples 2 \
  --revise 1 --plan --coder none --tag fresh_v15 > data/logs/fresh_v15.log 2>&1
echo DONE > data/logs/final_v15.done
