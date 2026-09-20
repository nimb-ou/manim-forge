#!/usr/bin/env bash
# Stop the worker pool and everything it started.
#
# SIGKILL on the supervisor skips its cleanup, and every child runs in its own
# process group (start_new_session=True, so a Ctrl-C in one terminal cannot
# take the pool down by accident). The consequence is that kill -9 leaves
# orphans that keep rendering -- eleven manim processes at 396% CPU, once,
# across four restarts nobody noticed.
#
# So: ask the supervisor to stop first and give it a moment, then sweep.
set -uo pipefail
cd "$(dirname "$0")/.."

pkill -TERM -f "scripts/forge_run.py" 2>/dev/null && sleep 4

for pat in "scripts/forge_run.py" "manim render" "scripts/generate_forever.py" \
           "scripts/generate_tasks.py" "scripts/render_showcase.py" \
           "scripts/regate.py" "scripts/verify_gold.py"; do
  pkill -9 -f "$pat" 2>/dev/null
done
sleep 1

left=$(pgrep -f "manim render|scripts/forge_run.py|scripts/generate_" | wc -l | tr -d ' ')
echo "forge stopped; $left related processes remain"
