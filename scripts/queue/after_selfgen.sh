#!/bin/bash
# After the self-training rows: measure the self-check on rows graded by eye.
cd "$(dirname "$0")/../.."
until grep -q DONE data/logs/selfgen.log 2>/dev/null; do sleep 60; done
.venv/bin/python -u scripts/measure_selfcheck.py > data/logs/selfcheck.log 2>&1
echo DONE >> data/logs/selfcheck.log
