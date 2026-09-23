#!/usr/bin/env python3
"""Reshape the planner's arcs into incremental windows.

    ./.venv/bin/python scripts/build_planner_windows.py

`build_planner_mix.py` produces one row per video: request -> the whole arc.
Tokenised against the real tokenizer those rows are a median of 3,620 tokens,
p90 8,310, max 32,888. At the 2048 the T4 can afford, **66% of them
truncate** -- so training on them would cut two thirds of the plans in half
and teach the planner to stop mid-arc, which is the exact length failure the
planner exists to fix.

A bigger context is the wrong fix: the logits tensor alone is 152k vocab x
4 bytes per position, and the scene run already spent five attempts fighting
a T4 for 2048.

So change the shape instead. A plan is generated left to right anyway, so
the unit is "given the request and the beats so far, write the next few" --
which fits in 2048 with room, and composes to any length at inference by
feeding its own output back. The 32,888-token arc becomes a dozen rows that
each fit, and none of them are truncated.

The cost is honest and worth stating: the model never sees a whole arc in
one window, so nothing in training teaches it where an arc should *end*. The
last window of each arc carries an explicit END marker so that is learnable
rather than left to luck.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRIDE = 6          # beats emitted per row
CONTEXT = 12        # beats of history shown (intents only)
SYSTEM = (
    "You plan 3Blue1Brown-style mathematical animations, a few beats at a "
    "time. You are given the request and the beats already planned. Continue "
    "the arc: write the next beats and nothing else, in the form\n"
    "  N. [seconds] intent -- narration\n"
    "keeping the numbering running. Build a real explanatory arc rather than "
    "a list of topics. When the explanation is complete, write END on its "
    "own line after the last beat."
)
LINE = re.compile(r"^\s*(\d+)\.\s*(.*)$")


def strip_narration(beat: str) -> str:
    """`N. [12s] intent -- narration` -> `N. [12s] intent`."""
    return beat.split(" -- ", 1)[0].rstrip()


def beats_of(plan: str) -> list[str]:
    return [l.rstrip() for l in plan.splitlines() if LINE.match(l)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "planner" / "plan.jsonl")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "planner" / "plan_windows.jsonl")
    a = ap.parse_args()

    rows, arcs = [], 0
    for line in a.src.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        request = r["messages"][1]["content"]
        beats = beats_of(r["messages"][2]["content"])
        if len(beats) < 3:
            continue
        arcs += 1
        for start in range(0, len(beats), STRIDE):
            window = beats[start:start + STRIDE]
            if not window:
                break
            # History carries intent only, not narration. With full
            # narration the windows were a median of 1,957 tokens and 32%
            # still truncated at 2048 -- sixteen beats of speech is about
            # 1,100 words, and most of it is spent restating what the model
            # only needs to know happened. The window being written keeps
            # its narration, because that is the thing being learned.
            history = [strip_narration(b)
                       for b in beats[max(0, start - CONTEXT):start]]
            shown = "\n".join(history) if history else \
                "  (nothing yet — open the explanation)"
            if start > CONTEXT:
                shown = f"  … {start - CONTEXT} earlier beats …\n" + shown
            last = start + STRIDE >= len(beats)
            answer = "\n".join(window) + ("\nEND" if last else "")
            rows.append({
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content":
                        f"REQUEST\n{request}\n\nBEATS SO FAR\n{shown}\n\n"
                        f"Write the next {len(window)} beat(s), numbered from "
                        f"{start + 1}."},
                    {"role": "assistant", "content": answer}],
                "meta": {**r["meta"],
                         "id": f"{r['meta']['id']}:w{start // STRIDE}",
                         "task": "plan-window", "window": start // STRIDE,
                         "final": last, "n_beats": len(window)},
            })

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"{len(rows)} windows from {arcs} arcs -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
