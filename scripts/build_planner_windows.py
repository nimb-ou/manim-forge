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
import hashlib
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


VARIANTS = ROOT / "data" / "planner" / "request_variants.jsonl"
JUDGED = ROOT / "data" / "planner" / "arc_judgements.jsonl"
FIXED = ROOT / "data" / "planner" / "plan_synth_fixed.jsonl"
FIXED_JUDGED = ROOT / "data" / "planner" / "arc_judgements_fixed.jsonl"


def _jsonl(path: Path) -> list[dict]:
    return ([json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            if path.exists() else [])


def intent_of(beat: str) -> str:
    """The beat's intent, keyed the way inference detects loops."""
    import sys
    sys.path.insert(0, str(ROOT))
    from forge.app.twostage import intent_key
    m = re.match(r"^\s*\d+\.\s*\[[^\]]*\]\s*(.*?)\s*(?:--|—|$)", beat)
    return intent_key(m.group(1) if m else beat)


def first_repeat_cut(beats: list[str]) -> list[str]:
    """The arc up to the first beat whose intent was already used.

    Synthetic arcs written to a fixed thirty-beat quota padded small
    requests by repeating one intent -- 3,425 duplicate intents across 259
    of 508 arcs -- and planner v1 duly wrote "The function graphed on a
    timeline" nine times running. Real arcs repeat an intent 58 times in
    5,984 beats, so this is applied to synthetic arcs only.
    """
    seen: set[str] = set()
    for i, b in enumerate(beats):
        k = intent_of(b)
        if k in seen:
            return beats[:i]
        seen.add(k)
    return beats


def pick_request(meta: dict, request: str, variants: dict) -> str:
    """The request at one of three lengths, fixed per arc.

    Corpus requests are paragraphs; the hard eval's median prompt is six
    words, and so is a person typing into the app. Where synth_requests.py
    has written shorter versions, 40% of arcs are asked tersely, 30% in one
    sentence, and 30% keep the paragraph. Hashed on the arc id, so every
    window of an arc sees the same request and a rebuild does not reshuffle.
    """
    v = variants.get(meta.get("id"))
    if not v:
        return request
    h = int(hashlib.sha256(meta["id"].encode()).hexdigest(), 16) % 10
    return v["short"] if h < 4 else v["sentence"] if h < 7 else request


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "planner" / "plan.jsonl")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "planner" / "plan_windows.jsonl")
    a = ap.parse_args()

    variants = ({json.loads(l)["id"]: json.loads(l)
                 for l in VARIANTS.read_text().splitlines() if l.strip()}
                if VARIANTS.exists() else {})
    # Arcs a second teacher judged mathematically wrong (judge_arcs.py).
    # An arc judged wrong is replaced by its corrected version (fix_arcs.py)
    # if a second judgement passed the correction, and dropped otherwise.
    wrong = {j["id"] for j in _jsonl(JUDGED) if j["verdict"] == "ERROR"}
    fixed_ok = {j["id"] for j in _jsonl(FIXED_JUDGED) if j["verdict"] == "OK"}
    fixed = {r["meta"]["id"]: r for r in _jsonl(FIXED)
             if r["meta"]["id"] in fixed_ok}
    replaced = 0
    rows, arcs, varied, cut_arcs, cut_beats = [], 0, 0, 0, 0
    judged_out = 0
    for line in a.src.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["meta"].get("id") in wrong:
            if r["meta"]["id"] in fixed:
                r = fixed[r["meta"]["id"]]
                replaced += 1
            else:
                judged_out += 1
                continue
        request = pick_request(r["meta"], r["messages"][1]["content"],
                               variants)
        beats = beats_of(r["messages"][2]["content"])
        if r["meta"].get("source") == "synthetic":
            full = len(beats)
            beats = first_repeat_cut(beats)
            if len(beats) < 6:
                cut_arcs += 1
                continue
            cut_beats += full - len(beats)
        if len(beats) < 3:
            continue
        arcs += 1
        varied += request != r["messages"][1]["content"]
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
            # The ask has to match what is being answered. A final window
            # holds whatever is left -- 3.8 beats on average -- and asking
            # "write the next 6" above an answer of 3 teaches the model that
            # the number requested is advisory. At inference the driver asks
            # for 6 every time, so if END only ever follows a short request
            # the model has no reason to produce one.
            answer = "\n".join(window) + ("\nEND" if last else "")
            rows.append({
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content":
                        f"REQUEST\n{request}\n\nBEATS SO FAR\n{shown}\n\n"
                        f"Write the next {STRIDE} beat(s), numbered from "
                        f"{start + 1}. Stop early and write END if the "
                        f"explanation is complete."},
                    {"role": "assistant", "content": answer}],
                "meta": {**r["meta"],
                         "id": f"{r['meta']['id']}:w{start // STRIDE}",
                         "task": "plan-window", "window": start // STRIDE,
                         "final": last, "n_beats": len(window)},
            })

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"{len(rows)} windows from {arcs} arcs -> {a.out}"
          + (f" ({varied} arcs asked with a shorter request)" if varied else ""))
    if judged_out or replaced:
        print(f"  judged wrong: {replaced} replaced by a checked correction, "
              f"{judged_out} dropped")
    if cut_arcs or cut_beats:
        print(f"  repeated intents: {cut_beats} beats cut, {cut_arcs} arcs "
              f"dropped (under six beats before the first repeat)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
