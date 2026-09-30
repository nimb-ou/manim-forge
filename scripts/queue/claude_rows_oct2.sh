#!/bin/zsh
# Rebuild the Claude rows (every teacher batch written by then) well before
# the Oct 3 00:30Z launch, so the launcher finds them up to date and pushes
# at once instead of rendering ~400 scenes first.
cd "$(dirname "$0")/../.." || exit 1
until [ "$(date -u +%Y%m%d%H%M)" -ge 202610021800 ]; do sleep 600; done
.venv/bin/python -u scripts/claude_kit_scenes.py > data/logs/claude_rows_oct2.log 2>&1
