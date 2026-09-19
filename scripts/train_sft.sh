#!/usr/bin/env bash
# LoRA supervised fine-tune on Apple Silicon via MLX.
#
# --mask-prompt matters more than it looks: without it the model is also trained
# to predict the *prompt*, which wastes capacity on text it will always be given
# and dilutes the signal we actually want, which is prompt -> code.
#
# --num-layers 16 over a 7B 4-bit base keeps resident memory near 6 GB, which
# leaves the machine usable. Raise it once we know the data is worth it.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-mlx-community/Qwen2.5-Coder-7B-Instruct-4bit}"
ITERS="${ITERS:-400}"
ADAPTER="${ADAPTER:-adapters/sft-$(date +%m%d-%H%M)}"

mkdir -p "$ADAPTER"
echo "model   : $MODEL"
echo "iters   : $ITERS"
echo "adapter : $ADAPTER"
echo

./.venv/bin/python -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data data/train \
  --fine-tune-type lora \
  --num-layers 16 \
  --batch-size 1 \
  --grad-accumulation-steps 4 \
  --iters "$ITERS" \
  --learning-rate 1e-5 \
  --max-seq-length 2048 \
  --mask-prompt \
  --steps-per-report 20 \
  --steps-per-eval 100 \
  --val-batches 12 \
  --save-every 100 \
  --grad-checkpoint \
  --adapter-path "$ADAPTER"

echo
echo "adapter written to $ADAPTER"
