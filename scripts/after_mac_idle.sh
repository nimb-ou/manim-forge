#!/bin/sh
# Usage: scripts/after_mac_idle.sh scripts/queue/<job>.sh  (a file, never an
# inline command: an inline command sits in this command line and matches.)
# Run a command once no MLX job (demo, two-stage, beat eval, prompt probe)
# is running. Queued commands live in the arguments of *this* script, whose
# own command line cannot match the pattern: a waiter that pgreps for text
# in its own `sh -c` string waits forever, which has happened three times.
cd "$(dirname "$0")/.." || exit 1
while pgrep -f "python[0-9.]* -u scripts/(demo|run_twostage|beat_eval|prompt_probe|scorecard|self_kit_beats|train_kit_coder)[.]py" >/dev/null; do
  sleep 30
done
exec "$@"
