#!/usr/bin/env bash
# Render gold scenes at presentation quality (1080p60) rather than the
# 480p15 used for gating. Gating optimises for throughput; gold scenes are
# the reference and should look like the product.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/Library/TeX/texbin:$PATH"
Q="${Q:--qh}"
for spec in "forge/gold/rook_corners.py RookCorners" \
            "forge/gold/gradient_descent.py LearningRate" \
            "forge/gold/dot_product.py DotProduct"; do
  set -- $spec
  echo "rendering $2 at $Q ..."
  ./.venv/bin/manim render "$Q" --disable_caching --save_sections \
      --media_dir data/renders "$1" "$2" 2>&1 | grep -E "Rendered|Error" || true
done
find data/renders -name "*.mp4" -path "*1080p60*" -not -path "*partial*" -exec ls -lh {} \; 2>/dev/null | awk '{print $5, $9}'
