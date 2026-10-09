#!/bin/bash
# v1.5 phase 5 data: the shipped engine (plan, rewrite) on 300 library
# requests, after the dev-set control has finished (one 9B at a time).
cd "$(dirname "$0")/../.."
until grep -q DONE data/logs/world_b2.log 2>/dev/null; do sleep 30; done
.venv/bin/python -u scripts/selfgen.py --n 300 > data/logs/selfgen.log 2>&1
echo DONE >> data/logs/selfgen.log
