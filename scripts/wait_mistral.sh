#!/bin/zsh
# Probe the Mistral teacher hourly; when its budget is back, unpause the
# kit shards and restart the supervisor so they run again (they gave up on
# 402s). One tiny request per hour.
cd "$(dirname "$0")/.." || exit 1
while true; do
  if .venv/bin/python - <<'P' 2>/dev/null; then
import sys; sys.path.insert(0, ".")
from forge.synth.teacher import Teacher
Teacher(provider="mistral", model="mistral-medium-latest").ask("Reply OK.", max_tokens=5)
P
    echo "$(date -u) mistral answers: restarting the kit shards"
    rm -f data/kit/pause_1 data/kit/pause_2
    pkill -f "scripts/supervisor.py"; sleep 3
    nohup .venv/bin/python -u scripts/supervisor.py --every 60 --retire-after 999999 \
      >> data/logs/supervisor_nohup.log 2>&1 < /dev/null &
    exit 0
  fi
  echo "$(date -u) still 402"
  sleep 3600
done
