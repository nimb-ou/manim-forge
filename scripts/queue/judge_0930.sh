#!/bin/zsh
# Gemini's daily quota resets ~07:00Z: judge today's runs with the headline
# judge (the local judge's +6 points on held-out need confirming).
cd "$(dirname "$0")/../.." || exit 1
until [ "$(date -u +%Y%m%d%H%M)" -ge 202609300710 ]; do sleep 300; done
while pgrep -f "scorecard.py" >/dev/null; do sleep 60; done
.venv/bin/python -u scripts/judge_sheets.py ab_held_scaf ab_short_scaf oracle_held > data/logs/judge_0930.log 2>&1
