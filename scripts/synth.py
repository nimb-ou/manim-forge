"""Generate synthetic training scenes with a teacher model, and gate them.

Generation and verification are interleaved on purpose. Every scene is rendered
the moment it is written, so the run reports a *verified* yield rather than a
raw count — and a prompt that produces plausible-looking code which never
renders is caught in the first minute, not after burning a day's quota.

Resumable: completed (topic, length) pairs are skipped on restart, so hitting a
daily rate limit costs nothing.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default=None, help="API model id; default per provider")
    ap.add_argument("--out", default="data/synthetic/generated.jsonl")
    ap.add_argument("--n", type=int, default=20, help="how many scenes to generate")
    ap.add_argument("--delay", type=float, default=4.0,
                    help="seconds between calls; free tier is ~15 req/min")
    ap.add_argument("--lengths", default="short,medium,long")
    a = ap.parse_args()

    from forge.synth.teacher import Teacher
    from forge.synth.topics import ALL, LENGTHS
    from forge.harness import RenderHarness
    from forge.repair.lint import lint
    from forge.ingest.schema import CorpusRow

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        for line in out.open():
            try:
                r = json.loads(line)
                done.add((r["topic"], r["length"]))
            except (json.JSONDecodeError, KeyError):
                continue

    wanted = {s.strip() for s in a.lengths.split(",")}
    jobs = [(t, d, ln, nb, hint) for (t, d) in ALL
            for (ln, nb, hint) in LENGTHS if ln in wanted
            and (t, ln) not in done]
    random.shuffle(jobs)          # spread domains across a rate-limited day
    jobs = jobs[: a.n]

    print(f"{len(done)} already generated | {len(jobs)} to do this run")
    if not jobs:
        return

    teacher = Teacher(provider=a.provider, model=a.model)
    harness = RenderHarness(python_bin="./.venv/bin/python",
                            cache_dir="data/frames", timeout=90)
    print(f"teacher: {teacher.model} via {a.provider}\n")

    n_ok = n_generated = 0
    with out.open("a") as sink:
        for i, (topic, domain, length, n_beats, hint) in enumerate(jobs, 1):
            t0 = time.monotonic()
            try:
                code = teacher.generate(topic, n_beats, hint)
            except Exception as e:
                print(f"  [{i:>3}/{len(jobs)}] API ERROR {type(e).__name__}: {str(e)[:90]}")
                time.sleep(a.delay * 2)
                continue

            code, rules = lint(code)
            r = harness.render(code, quality="low", frames=4)
            row = CorpusRow.build(source=f"synth-{a.provider}", license="CC-BY-NC-SA-4.0",
                                  prompt=topic, code=code, index=i,
                                  tags=[f"domain:{domain}", f"length:{length}", "synthetic"])
            rec = row.to_dict()
            rec.update({"topic": topic, "domain": domain, "length": length,
                        "ok": r.ok, "error_kind": r.error_kind.value,
                        "n_frames": len(r.frame_paths), "video_s": r.duration_s,
                        "lint": rules})
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            n_ok += r.ok
            n_generated += 1

            flag = "PASS" if r.ok else f"FAIL {r.error_kind.value}"
            print(f"  [{i:>3}/{len(jobs)}] {flag:<22} {domain:<16} beats={row.n_play_calls:<3} "
                  f"{time.monotonic()-t0:4.1f}s  | {topic[:44]}", flush=True)

            time.sleep(max(0.0, a.delay - (time.monotonic() - t0)))

    print(f"\n{'='*58}")
    # API errors say nothing about the teacher, exactly as environment
    # failures say nothing about a corpus row. Scoring them as failures
    # would make a rate limit look like bad data.
    api_errors = len(jobs) - n_generated
    denom = n_generated or 1
    print(f"  generated        : {n_generated}/{len(jobs)}  ({api_errors} API errors, excluded)")
    print(f"  verified yield   : {n_ok}/{n_generated} = {n_ok/denom:.1%}")
    print(f"  written to {out}")
    print("="*58)


if __name__ == "__main__":
    main()
