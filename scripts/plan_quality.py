#!/usr/bin/env python3
"""Compare teacher-written arcs against the real 3Blue1Brown ones.

    ./.venv/bin/python scripts/plan_quality.py

`synth_plans.py` is producing arcs to grow the planner's 190 rows, and the
plan says they enter at a lower weight because a teacher's imitation is
worth less than the thing imitated. That is an assumption, and this is the
cheapest way to check it before a GPU session is spent on it.

Nothing here judges whether an arc is *good* -- no model scores another
model. It measures the things a real arc demonstrably has, where the real
arcs are 146 actual videos with true timings:

  beats        how many steps an explanation takes
  seconds      real beats run about 26s; the untuned model guessed 0, 2, 4
  narration    words per beat, and whether it reads as speech or as a
               table of contents ("Introduce the concept of ...")
  intent       whether the visual is named at all

A synthetic arc that matches on pacing and diverges on narration is still
worth training on for structure. One that diverges on both is worth less
than its weight, and the answer is to stop generating rather than to
generate more.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEAT = re.compile(r"^\s*(\d+)\.\s*(?:\[([^\]]*)\])?\s*(.*?)(?:\s+--\s+(.*))?$")
# Phrases that describe an explanation instead of being one. The untuned
# planner produced these and nothing else, so they are the failure mode this
# is watching for coming back in through the teacher.
CONTENTS = re.compile(
    r"\b(?:introduce|explain(?:s|ing)? (?:that|how|the)|describe|discuss|"
    r"outline|cover|present|demonstrate) (?:the|a|an|that|how)\b", re.I)


def arcs(path: Path, source: str | None = None) -> list[list[dict]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if source and r["meta"].get("source") != source:
            continue
        beats = []
        for l in r["messages"][-1]["content"].splitlines():
            m = BEAT.match(l)
            if not m or not (m.group(3) or "").strip():
                continue
            secs = None
            try:
                secs = float((m.group(2) or "").strip().rstrip("s"))
            except ValueError:
                pass
            beats.append({"seconds": secs, "intent": m.group(3).strip(),
                          "narration": (m.group(4) or "").strip()})
        if beats:
            out.append(beats)
    return out


def describe(name: str, data: list[list[dict]]) -> dict:
    if not data:
        return {}
    flat = [b for arc in data for b in arc]
    secs = [b["seconds"] for b in flat if b["seconds"]]
    words = [len(b["narration"].split()) for b in flat if b["narration"]]
    contents = sum(1 for b in flat if CONTENTS.search(b["narration"]))
    no_intent = sum(1 for b in flat if not b["intent"])
    d = {
        "arcs": len(data),
        "beats/arc": round(st.mean(len(a) for a in data), 1),
        "seconds/beat": round(st.mean(secs), 1) if secs else 0.0,
        "sec p10-p90": (f"{sorted(secs)[len(secs)//10]:.0f}-"
                        f"{sorted(secs)[9*len(secs)//10]:.0f}" if secs else "-"),
        "words/beat": round(st.mean(words), 1) if words else 0.0,
        # The internal check: does the narration fill the duration the beat
        # claims? A beat saying [23s] with 41 words is asking for 23 seconds
        # of screen time and supplying 12 seconds of speech. That is not a
        # style difference from the real arcs, it is the arc contradicting
        # itself, and it is invisible in either number alone.
        "words/sec": round(
            st.mean([len(b["narration"].split()) / b["seconds"]
                     for b in flat if b["seconds"] and b["narration"]]), 2)
        if secs else 0.0,
        "table-of-contents %": round(100 * contents / len(flat), 1),
        "no intent %": round(100 * no_intent / len(flat), 1),
    }
    print(f"{name:14s} " + "  ".join(f"{k} {v}" for k, v in d.items()))
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", type=Path,
                    default=ROOT / "data" / "planner" / "plan_synth.jsonl")
    a = ap.parse_args()

    plan = ROOT / "data" / "planner" / "plan.jsonl"
    real = describe("real (3b1b)", arcs(plan, "narration"))
    gold = describe("gold", arcs(plan, "gold"))
    synth = describe("synthetic", arcs(a.synth))

    if not (real and synth):
        print("\nnot enough data to compare yet")
        return 0

    print("\nverdict:")
    for key, tol, why in (
        ("seconds/beat", 0.45, "pacing — what a beat costs"),
        ("words/beat", 0.50, "narration density"),
        ("words/sec", 0.35, "does the narration fill the time it asks for"),
    ):
        r, s = real[key], synth[key]
        off = abs(s - r) / max(r, 1e-9)
        mark = "ok  " if off <= tol else "OFF "
        print(f"  {mark} {key:14s} real {r} vs synthetic {s} "
              f"({off:+.0%}) — {why}")
    toc = synth["table-of-contents %"]
    print(f"  {'ok  ' if toc < 10 else 'OFF '} table-of-contents "
          f"{toc}% of synthetic beats vs {real['table-of-contents %']}% real "
          f"— describing rather than narrating")
    print("\nA synthetic arc matching on pacing and diverging on narration is "
          "still\nworth training on for structure. Diverging on both is worth "
          "less than\nits weight, and the answer is to stop generating.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
