#!/usr/bin/env python3
"""Write beat arcs for corpus topics, in the style of the real ones.

    ./.venv/bin/python scripts/synth_plans.py

The planner has 190 arcs: 44 gold and 146 real 3Blue1Brown videos. That is
everything the scrape contains, and it is a small training set for a task
this open-ended.

The corpus has 1,516 verified scene requests with no arcs at all. A teacher
can write one for each -- and unlike generating more *scenes*, this does not
risk more slideware, because a plan is never rendered and never trains the
coder. The worst a bad plan does is teach a bad arc, which is visible by
reading it.

**Every generated arc is shown a real one first.** The formatting spec alone
produced "1. 0.000 intent -- Introduce the concept of light", which is a
table of contents, not narration. A real arc in the prompt is the difference
between imitating 3Blue1Brown and describing him.

Output is a separate file and a separate tier. Mixing teacher arcs into the
same weight as 146 real ones would make the planner's next number
uninterpretable -- which of the two taught it is exactly what has to stay
answerable.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (
    "You plan 3Blue1Brown-style mathematical animations. Produce ONLY a "
    "numbered list of beats, each as\n"
    "  N. [seconds] intent -- narration\n"
    "where intent is what is on screen and narration is what is spoken over "
    "it. Beats run 15-30 seconds. Build a real explanatory arc: establish, "
    "develop, complicate, resolve. Write END on its own line after the last "
    "beat. Write no code and no commentary."
)
PROMPT = (
    "Here is a real arc, for style and pacing:\n\n{example}\n\n"
    "Now write an arc of {n} beats for this request:\n{request}"
)
LINE = re.compile(r"^\s*\d+\.\s")


def real_arcs() -> list[str]:
    path = ROOT / "data" / "planner" / "plan.jsonl"
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        beats = [l for l in r["messages"][2]["content"].splitlines()
                 if LINE.match(l)]
        if 6 <= len(beats) <= 14:
            out.append("\n".join(beats[:10]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default="gemini-3.7-flash")
    ap.add_argument("--beats", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "planner" / "plan_synth.jsonl")
    a = ap.parse_args()

    from forge.synth.teacher import Teacher
    teacher = Teacher(provider=a.provider, model=a.model)
    examples = real_arcs()
    if not examples:
        print("no real arcs to imitate; run build_planner_mix.py first")
        return 1

    # Corpus prompts carry the scraper's instruction wrapper -- "Generate
    # accurate and correct ManimCE Python code for the animation requested by
    # the user. Here is the user's ..." -- which is an instruction to a code
    # model and has nothing to do with planning an explanation. Left in, the
    # planner would learn to plan arcs *about writing Manim code*.
    WRAPPER = re.compile(
        r"^.*?(?:here is the user'?s?[^:]*:|user'?s? request:?)\s*", re.I | re.S)
    requests = []
    for line in (ROOT / "data" / "verified" / "example_index.jsonl") \
            .read_text().splitlines():
        if not line.strip():
            continue
        req = WRAPPER.sub("", json.loads(line)["prompt"].strip()).strip()
        if len(req) > 40:
            requests.append(req)
    # Longest requests first: a one-line "draw a circle" has no arc in it,
    # and spending teacher calls on those buys nothing.
    requests = sorted(set(requests), key=len, reverse=True)[: a.limit or None]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.out.exists():
        for line in a.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["meta"]["request"])
        print(f"resuming: {len(done)} arcs already written", flush=True)

    rng = random.Random(17)
    made = failed = 0
    for n, req in enumerate(requests, 1):
        if req in done:
            continue
        try:
            reply = teacher.ask(
                PROMPT.format(example=rng.choice(examples), n=a.beats,
                              request=req),
                max_tokens=2000, system=SYSTEM)
        except Exception as exc:                              # noqa: BLE001
            failed += 1
            print(f"  [{n}/{len(requests)}] {type(exc).__name__}", flush=True)
            time.sleep(a.pause * 3)
            continue
        beats = [l.rstrip() for l in reply.splitlines() if LINE.match(l)]
        if len(beats) < 5:
            failed += 1
            print(f"  [{n}/{len(requests)}] {len(beats)} beats, skipped",
                  flush=True)
            time.sleep(a.pause)
            continue
        with a.out.open("a") as f:
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": req},
                    {"role": "assistant", "content": "\n".join(beats) + "\nEND"}],
                "meta": {"id": f"plan-synth:{n:05d}", "source": "synthetic",
                         "task": "plan", "n_beats": len(beats),
                         "request": req, "teacher": f"{a.provider}:{a.model}"},
            }) + "\n")
        done.add(req)
        made += 1
        if made % 10 == 0 or made < 4:
            print(f"  [{n}/{len(requests)}] {made} arcs, {failed} failed",
                  flush=True)
        time.sleep(a.pause)

    print(f"\n{made} arcs written, {failed} failed -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
