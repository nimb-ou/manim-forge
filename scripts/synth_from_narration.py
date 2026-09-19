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

INSTRUCTION = """Below is a passage of narration from a mathematics video.

Write the Manim scene that should be on screen while this is being said.
Follow every rule you were given about beats, computed values and layout.
Set each beat's narration= to the part of the passage it covers.

If the passage cannot sensibly be animated — it is an aside, a sponsor
message, a reference to another video, or pure commentary with no visual
content — reply with exactly SKIP and nothing else.

PASSAGE:
"""


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
    from forge.repair.lint import lint
    from forge.ingest.schema import CorpusRow

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
            try:
                code = teacher.generate(INSTRUCTION + s["text"], 6, "matching the passage")
            except Exception as e:
                print(f"  [{i:>3}/{len(todo)}] API {type(e).__name__}", flush=True)
                time.sleep(a.delay * 2)
                continue
            n_gen += 1

            if code.strip().upper().startswith("SKIP") or len(code) < 120:
                n_skip += 1
                print(f"  [{i:>3}/{len(todo)}] SKIP (not animatable)  {s['title'][:40]}", flush=True)
                continue

            code, rules = lint(code)
            r = harness.render(code, quality="low", frames=4)
            row = CorpusRow.build(source="synth-narration", license="CC-BY-NC-SA-4.0",
                                  prompt=s["text"], code=code, index=i,
                                  tags=["synthetic", "from-3b1b-narration"])
            rec = row.to_dict()
            rec.update({"video_id": s["video_id"], "seg_index": s["index"],
                        "title": s["title"], "ok": r.ok,
                        "error_kind": r.error_kind.value, "lint": rules,
                        "video_s": r.duration_s,
                        "teacher_model": getattr(teacher, "last_model_used", "")})
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            n_ok += r.ok
            flag = "PASS" if r.ok else f"FAIL {r.error_kind.value}"
            print(f"  [{i:>3}/{len(todo)}] {flag:<20} {time.monotonic()-t0:4.1f}s  {s['title'][:40]}", flush=True)
            time.sleep(max(0.0, a.delay - (time.monotonic() - t0)))

    scored = n_gen - n_skip
    print(f"\n{'='*58}")
    print(f"  generated {n_gen} | skipped as non-visual {n_skip}")
    print(f"  verified  {n_ok}/{scored} = {n_ok/max(scored,1):.1%}")
    print("="*58)


if __name__ == "__main__":
    main()
