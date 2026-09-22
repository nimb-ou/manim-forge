#!/usr/bin/env python3
"""Build training rows for the *planner*, which has never been trained.

    ./.venv/bin/python scripts/build_planner_mix.py                 # gold only
    ./.venv/bin/python scripts/build_planner_mix.py --with-intents  # + 151 arcs

**Why a planner at all.** Run 17's failure is ambition outrunning API
competence inside one generation: it plans well now (beats 0 -> 3.29,
coverage 2x, length 2x) and then has to implement all of that in a single
pass, where it reaches for `Polyhedron(vertex_coords=...)` instead of
`Dodecahedron()`. Its failures are 80% longer than its own passes.

Splitting the job makes the coder's task smaller rather than making the
model better at a task that is too big -- and **ambition expressed in text
cannot fail to render**. The architecture has had a planner in it since the
start (`forge/synth/tasks.py` defines PLAN and DECOMPOSE), but all 3,010
rows of the training mix are prompt->code. The planner has never existed.

**Two sources, both already on disk.**

*Gold scenes.* 182 `@beat(...)` decorators across 44 scenes, each carrying
intent, seconds and narration. Exact supervision, tiny.

*Real narration.* 5,825 segments across 151 actual 3Blue1Brown videos, with
true timings -- median 35 segments of 26s, which is a 15-minute arc. This is
the only thing in the project that knows what a 16-minute explanation is
shaped like, and the hard eval asks for 16-minute explanations and gets 3.5%
of one. What the segments lack is *intent*: what is on screen. One API call
per video fills that in, 151 calls rather than 5,825, because the arc has to
be read as a whole anyway.

One output format for both, or the adapter learns two:

    N. [seconds] intent -- narration
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 12
SYSTEM = (
    "You plan 3Blue1Brown-style mathematical animations. Given a request, "
    "produce ONLY a numbered list of beats. A beat is one communicative step: "
    "something appears, transforms, or is annotated. Each line is\n"
    "  N. [seconds] intent -- narration\n"
    "where intent is what happens on screen and narration is what is said "
    "over it. Build a real explanatory arc: establish, develop, complicate, "
    "resolve. Write no code."
)


def beats_of(source: str) -> list[dict]:
    """Pull @beat(...) decorators out of a gold scene, in source order."""
    out = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            name = dec.func.id if isinstance(dec.func, ast.Name) else \
                getattr(dec.func, "attr", "")
            if name != "beat":
                continue
            intent = (dec.args[0].value
                      if dec.args and isinstance(dec.args[0], ast.Constant)
                      else "")
            kw = {k.arg: k.value for k in dec.keywords}
            secs = (kw["seconds"].value
                    if "seconds" in kw and isinstance(kw["seconds"], ast.Constant)
                    else None)
            narr = (kw["narration"].value
                    if "narration" in kw and isinstance(kw["narration"], ast.Constant)
                    else "")
            out.append({"intent": intent, "seconds": secs,
                        "narration": " ".join(str(narr).split()),
                        "lineno": dec.lineno})
    return sorted(out, key=lambda b: b["lineno"])


# Subtitle files carry corrections boilerplate, and 3Blue1Brown videos end
# with a patron thank-you that is not part of the explanation. Both would be
# learned as the shape of an explainer: open with a URL, close by thanking
# people.
BOILERPLATE = re.compile(r"\[[^\]]*(?:subtitle|caption|correction)[^\]]*\]",
                         re.I)
OUTRO = re.compile(r"patreon|patron|sponsor|thanks for watching|subscribe|"
                   r"supporters|brilliant\.org", re.I)


def clean(beats: list[dict]) -> list[dict]:
    """Strip subtitle boilerplate, and drop the outro off the tail.

    Only from the *tail*: a video may mention a sponsor mid-way while still
    explaining something, and cutting from the middle would leave a plan
    whose timings no longer describe a continuous arc.
    """
    for b in beats:
        b["narration"] = " ".join(BOILERPLATE.sub("", b["narration"]).split())
    end = len(beats)
    while end > 1 and OUTRO.search(beats[end - 1]["narration"]):
        end -= 1
    return [b for b in beats[:end] if b["narration"]]


def render_plan(beats: list[dict]) -> str:
    lines = []
    for i, b in enumerate(beats, 1):
        secs = f"{b['seconds']:g}s" if b.get("seconds") else "?"
        intent = b["intent"] or "(unspecified)"
        lines.append(f"{i}. [{secs}] {intent} -- {b['narration']}".rstrip(" -"))
    return "\n".join(lines)


def row(request: str, plan: str, meta: dict) -> dict:
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": request},
                         {"role": "assistant", "content": plan}],
            "meta": meta}


def from_gold() -> list[dict]:
    path = ROOT / "data" / "gold" / "gold.jsonl"
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        beats = beats_of(g["code"])
        if len(beats) < 2:
            continue
        rows.append(row(g["prompt"], render_plan(beats),
                        {"id": f"plan-gold:{g['meta']['scene']}",
                         "source": "gold", "task": "plan",
                         "n_beats": len(beats)}))
    return rows


def arcs() -> list[tuple[str, list[dict]]]:
    """Group the narration corpus into one timed arc per video."""
    path = ROOT / "data" / "style" / "narration.jsonl"
    by_video: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by_video.setdefault(r["video_id"], []).append(r)
    out = []
    for vid, segs in by_video.items():
        segs.sort(key=lambda s: s["index"])
        out.append((segs[0]["title"], segs))
    return out


INTENT_PROMPT = (
    "Below is the narration of a 3Blue1Brown video, in order, with timings.\n"
    "For each numbered line, write what is most likely ON SCREEN during it — "
    "the visual, not a summary of the words. Six words or fewer.\n\n"
    "Answer with exactly one line per input line, numbered the same way, and "
    "nothing else.\n\nTITLE: {title}\n\n{body}"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "planner" / "plan.jsonl")
    ap.add_argument("--with-intents", action="store_true",
                    help="call the teacher to infer on-screen intent for the "
                         "151 narration arcs (one call per video)")
    ap.add_argument("--limit", type=int, default=0, help="cap arcs, for a trial")
    a = ap.parse_args()

    rows = from_gold()
    print(f"gold: {len(rows)} plan rows, "
          f"{sum(r['meta']['n_beats'] for r in rows)} beats")

    if a.with_intents:
        from forge.app.generator import RemoteGemini
        gen = RemoteGemini()
        todo = arcs()[: a.limit or None]
        print(f"narration: {len(todo)} arcs, one call each")
        for n, (title, segs) in enumerate(todo, 1):
            # Chunked. A whole arc is ~35 segments and about 12k input
            # tokens, and both trial calls hit the teacher's 240s timeout.
            # Twelve at a time still gives enough surrounding context for
            # "what is on screen here" to be answerable, and returns in
            # seconds.
            intents, failed = {}, False
            for start in range(0, len(segs), CHUNK):
                part = segs[start:start + CHUNK]
                body = "\n".join(
                    f"{start + j}. [{s['end'] - s['start']:.0f}s] "
                    f"{' '.join(s['text'].split())}"
                    for j, s in enumerate(part, 1))
                try:
                    reply = gen.complete(
                        "You describe what is on screen during a narrated "
                        "mathematical animation. Terse, visual, no commentary.",
                        INTENT_PROMPT.format(title=title, body=body),
                        max_tokens=1024)
                except Exception as exc:                      # noqa: BLE001
                    print(f"  [{n}/{len(todo)}] {title[:40]}: "
                          f"{type(exc).__name__} on chunk {start}", flush=True)
                    failed = True
                    break
                for line in reply.splitlines():
                    m = re.match(r"\s*(\d+)[.)]\s*(.+)", line)
                    if m:
                        intents[int(m.group(1))] = m.group(2).strip()
            if failed:
                continue
            if len(intents) < len(segs) * 0.6:
                print(f"  [{n}/{len(todo)}] {title[:40]}: only "
                      f"{len(intents)}/{len(segs)} intents, skipped", flush=True)
                continue
            beats = [{"intent": intents.get(i, ""),
                      "seconds": round(s["end"] - s["start"]),
                      "narration": " ".join(s["text"].split())}
                     for i, s in enumerate(segs, 1)]
            beats = clean(beats)
            if len(beats) < 4:
                print(f"  [{n}/{len(todo)}] {title[:40]}: "
                      f"{len(beats)} beats after cleaning, skipped", flush=True)
                continue
            rows.append(row(
                f"Explain, in the style of 3Blue1Brown: {title}",
                render_plan(beats),
                {"id": f"plan-narration:{segs[0]['video_id']}",
                 "source": "narration", "task": "plan", "n_beats": len(beats)}))
            print(f"  [{n}/{len(todo)}] {title[:50]} -> {len(beats)} beats",
                  flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    tot = sum(r["meta"]["n_beats"] for r in rows)
    print(f"\n{len(rows)} plan rows, {tot} beats -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
