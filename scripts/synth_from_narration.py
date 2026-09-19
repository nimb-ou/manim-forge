"""Generate scenes from 3Blue1Brown's own narration.

Our topic-list prompts read like a topic list. Real narration reads like a
person explaining something, which is both closer to what a user will type and
a direct sample of the style we are trying to learn.

So each segment becomes a prompt: *this is what the narrator is saying — build
the animation that should be on screen while they say it*. The pairing we get
is (explanation in his voice -> animation), which is exactly the planner's job.

Not every segment is animatable. Plenty are asides, sponsor reads, or
references to the previous video. Rather than guess with keyword heuristics,
the teacher is told it may reply SKIP, and those are dropped. A model deciding
"can this be drawn" is more reliable than a word list, and it costs one call we
were making anyway.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from forge.synth.teacher import NARRATION_INSTRUCTION as INSTRUCTION


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--narration", default="data/style/narration.jsonl")
    ap.add_argument("--out", default="data/synthetic/from_narration.jsonl")
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--min-chars", type=int, default=260)
    ap.add_argument("--max-chars", type=int, default=1600)
    a = ap.parse_args()

    from forge.synth.teacher import Teacher
    from forge.harness import RenderHarness
    from forge.ingest.schema import CorpusRow
    from forge.synth.repair_gen import generate_and_repair

    segs = [json.loads(l) for l in Path(a.narration).open()]
    # Very short segments carry no content; very long ones span several ideas
    # and produce sprawling scenes.
    segs = [s for s in segs if a.min_chars <= len(s["text"]) <= a.max_chars]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            try:
                r = json.loads(line)
                done.add((r["video_id"], r["seg_index"]))
            except (json.JSONDecodeError, KeyError):
                pass

    todo = [s for s in segs if (s["video_id"], s["index"]) not in done]
    random.Random(7).shuffle(todo)
    todo = todo[: a.n]
    print(f"{len(segs)} usable segments | {len(done)} done | {len(todo)} this run\n", flush=True)

    teacher = Teacher(provider="gemini", model=a.model)
    harness = RenderHarness(python_bin="./.venv/bin/python",
                            cache_dir="data/frames", timeout=90)

    n_ok = n_gen = n_skip = 0
    with out.open("a") as sink:
        for i, s in enumerate(todo, 1):
            t0 = time.monotonic()
            request = INSTRUCTION + s["text"]
            try:
                # Probe first: a SKIP costs one call, and repairing a scene the
                # teacher declined to write would be repairing nothing.
                code = teacher.generate(request, 6, "matching the passage")
            except Exception as e:
                print(f"  [{i:>3}/{len(todo)}] API {type(e).__name__}", flush=True)
                time.sleep(a.delay * 2)
                continue
            n_gen += 1

            if code.strip().upper().startswith("SKIP") or len(code) < 120:
                n_skip += 1
                print(f"  [{i:>3}/{len(todo)}] SKIP (not animatable)  {s['title'][:40]}", flush=True)
                continue

            g = generate_and_repair(teacher, harness, request, 6,
                                    "matching the passage", max_rounds=2,
                                    first_code=code)
            row = CorpusRow.build(source="synth-narration", license="CC-BY-NC-SA-4.0",
                                  prompt=s["text"], code=g.code, index=i,
                                  tags=["synthetic", "from-3b1b-narration"])
            rec = row.to_dict()
            rec.update({"video_id": s["video_id"], "seg_index": s["index"],
                        "title": s["title"], "ok": g.ok,
                        "error_kind": g.error_kind, "lint": g.lint_rules,
                        "repair_rounds": g.rounds, "history": g.history,
                        "video_s": g.duration_s,
                        "teacher_model": getattr(teacher, "last_model_used", "")})
            r = g
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            n_ok += r.ok
            flag = "PASS" if g.ok else f"FAIL {g.error_kind}"
            fixed = f" (fixed in {g.rounds})" if g.ok and g.rounds else ""
            print(f"  [{i:>3}/{len(todo)}] {flag:<20}{fixed:<14} "
                  f"{time.monotonic()-t0:5.1f}s  {s['title'][:40]}", flush=True)
            time.sleep(max(0.0, a.delay - (time.monotonic() - t0)))

    scored = n_gen - n_skip
    print(f"\n{'='*58}")
    print(f"  generated {n_gen} | skipped as non-visual {n_skip}")
    print(f"  verified  {n_ok}/{scored} = {n_ok/max(scored,1):.1%}")
    print("="*58)


if __name__ == "__main__":
    main()
