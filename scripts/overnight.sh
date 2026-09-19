#!/usr/bin/env bash
# Unattended supervisor.
#
# The daemons already survive rate limits and resume from disk. What they do
# not survive is dying — a crashed process stays dead, and ten hours of silence
# looks identical to ten hours of work. So this restarts them, guards the disk,
# and snapshots to Hugging Face periodically.
#
# Everything it supervises is resumable, so a restart costs at most one task.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/Library/TeX/texbin:$PATH"

LOG=data/logs/overnight.log
MIN_FREE_GB=8          # stop generating below this; renders need scratch space
BACKUP_EVERY=7200      # seconds

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

free_gb() { df -g /System/Volumes/Data | tail -1 | awk '{print $4}'; }

alive() { pgrep -f "$1" >/dev/null 2>&1; }

start_code() {
  nohup ./.venv/bin/python -u scripts/generate_forever.py \
      --max-hours 12 --workers 5 >> data/logs/stream.log 2>&1 &
  say "started code daemon (pid $!)"
}

start_prose() {
  nohup ./.venv/bin/python -u scripts/generate_tasks.py \
      --per-kind 400 --workers 3 --max-hours 12 >> data/logs/tasks.log 2>&1 &
  say "started prose daemon (pid $!)"
}

say "=== overnight supervisor up ==="
say "disk: $(free_gb) GB free"

last_backup=$SECONDS

while true; do
  free=$(free_gb)

  if [ "$free" -lt "$MIN_FREE_GB" ]; then
    say "DISK LOW (${free} GB) — pausing generation"
    pkill -f generate_forever; pkill -f generate_tasks
    sleep 600
    continue
  fi

  alive "generate_forever.py" || { say "code daemon down — restarting"; start_code; }
  alive "generate_tasks.py"   || { say "prose daemon down — restarting"; start_prose; }

  if [ $((SECONDS - last_backup)) -ge "$BACKUP_EVERY" ]; then
    say "snapshotting to Hugging Face ..."
    ./.venv/bin/python scripts/backup_to_hf.py >> "$LOG" 2>&1 \
      && say "snapshot ok" || say "snapshot FAILED (continuing)"
    last_backup=$SECONDS
  fi

  # A heartbeat with real counts, so ten hours of silence is distinguishable
  # from ten hours of work.
  ./.venv/bin/python - >> "$LOG" 2>&1 <<'PY'
import json, time
from pathlib import Path
def n(p, key=None):
    f = Path(p)
    if not f.exists(): return 0, 0
    rows = [json.loads(l) for l in f.open()]
    real = [r for r in rows if not r.get("skipped")]
    return sum(bool(r.get(key or "ok")) for r in real), len(real)
c_ok, c_n = n("data/synthetic/stream.jsonl")
p_ok, p_n = n("data/synthetic/tasks.jsonl", "valid")
print(f"[{time.strftime('%H:%M:%S')}] code {c_ok}/{c_n}  prose {p_ok}/{p_n}")
PY

  sleep 300
done
