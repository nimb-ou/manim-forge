#!/bin/bash
# Final test, v1.0 side: the v1.0 tag's own code (a worktree) on the 20
# fresh prompts, as the v1.0 app ran (release library, kit API, best of 2).
cd "/Users/nimitjain/Desktop/My Mac/Projects/manim-forge"
until grep -q "^B: \|Traceback" data/logs/selfcheck.log; do sleep 30; done
cd "/private/tmp/claude-501/-Users-nimitjain-Desktop-My-Mac-Projects-manim-forge/957f5636-529a-4635-a29c-741bbe21b713/scratchpad/v10"
FORGE_LIBRARY=release "/Users/nimitjain/Desktop/My Mac/Projects/manim-forge/.venv/bin/python" -u scripts/scorecard.py --inscope --n 20 --oneshot --api --coder none --samples 2 --tag fresh_v10 > "/Users/nimitjain/Desktop/My Mac/Projects/manim-forge/data/logs/fresh_v10.log" 2>&1
echo DONE >> "/Users/nimitjain/Desktop/My Mac/Projects/manim-forge/data/logs/fresh_v10.log"
