"""Run generation until the free tier is genuinely exhausted, then wait for it.

Rate limits are per model, so the daily budget is the sum across every model
that answers. This rotates through them, retires one on a 429, and sleeps
rather than exits when all are cooling down — a quota that resets in fifteen
minutes is worth waiting for on a job that has all night.

Safe to stop at any time: every result is written and flushed as it completes,
and completed task keys are skipped on restart.
"""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

STOP = False


def _handle(signum, frame):
    global STOP
    STOP = True
    print("\n  stop requested — finishing current task, then exiting", flush=True)


def main() -> None:
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic/stream.jsonl")
    ap.add_argument("--max-tasks", type=int, default=100000)
    ap.add_argument("--max-hours", type=float, default=24.0)
    ap.add_argument("--variations", type=int, default=1)
    ap.add_argument("--repair-rounds", type=int, default=2)
    ap.add_argument("--cooldown", type=float, default=900.0)
    a = ap.parse_args()

    from forge.synth.teacher import Teacher, GEMINI_ROTATION
    from forge.synth.daemon import ModelPool, build_queue, load_done
    from forge.synth.repair_gen import generate_and_repair
    from forge.harness import RenderHarness
    from forge.ingest.schema import CorpusRow

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)

    queue = [t for t in build_queue(variations=a.variations) if t.key not in done]
    print(f"queue: {len(queue)} tasks pending ({len(done)} already done)")
    print(f"models: {', '.join(GEMINI_ROTATION)}\n", flush=True)

    pool = ModelPool(list(GEMINI_ROTATION), cooldown_s=a.cooldown)
    harness = RenderHarness(python_bin="./.venv/bin/python",
                            cache_dir="data/frames", timeout=90)

    t_start = time.monotonic()
    n_done = n_ok = n_repaired = 0

    with out.open("a") as sink:
        for task in queue[: a.max_tasks]:
            if STOP or (time.monotonic() - t_start) / 3600 > a.max_hours:
                break

            model = pool.next_model()
            while model is None:
                wait = pool.seconds_until_any()
                print(f"  all models cooling down — sleeping {wait/60:.0f}m "
                      f"[{pool.status()}]", flush=True)
                slept = 0.0
                while slept < wait and not STOP:
                    time.sleep(min(20.0, wait - slept))
                    slept += 20.0
                if STOP:
                    break
                model = pool.next_model()
            if model is None:
                break

            teacher = Teacher(provider="gemini", model=model)
            t0 = time.monotonic()
            try:
                g = generate_and_repair(teacher, harness, task.request,
                                        task.n_beats, task.length_hint,
                                        max_rounds=a.repair_rounds)
            except Exception as e:
                msg = str(e)
                if "429" in msg or "503" in msg:
                    pool.retire(model)
                    print(f"  {model} exhausted -> cooldown [{pool.status()}]", flush=True)
                else:
                    print(f"  error {type(e).__name__}: {msg[:70]}", flush=True)
                continue

            if g.code.strip().upper().startswith("SKIP") or len(g.code) < 120:
                sink.write(json.dumps({"task_key": task.key, "kind": task.kind,
                                       "skipped": True}) + "\n")
                sink.flush()
                n_done += 1
                continue

            row = CorpusRow.build(
                source=f"stream-{task.kind}", license="CC-BY-NC-SA-4.0",
                prompt=task.request if task.kind == "topic" else task.request[-1400:],
                code=g.code, index=n_done,
                tags=["synthetic", task.kind] + [f"domain:{task.meta.get('domain','?')}"])
            rec = row.to_dict()
            rec.update({"task_key": task.key, "kind": task.kind,
                        "ok": g.ok, "error_kind": g.error_kind,
                        "repair_rounds": g.rounds, "history": g.history,
                        "teacher_model": getattr(teacher, "last_model_used", model),
                        "video_s": g.duration_s, **task.meta})
            sink.write(json.dumps(rec) + "\n")
            sink.flush()

            n_done += 1
            n_ok += g.ok
            n_repaired += 1 if (g.ok and g.rounds) else 0

            if n_done % 5 == 0 or g.ok:
                rate = n_ok / max(n_done, 1)
                el = (time.monotonic() - t_start) / 60
                flag = "PASS" if g.ok else f"FAIL {g.error_kind}"
                print(f"  [{n_done:>5}] {flag:<20} {task.kind:<9} "
                      f"yield {rate:5.1%}  {el:5.1f}m  {model.replace('gemini-','')}",
                      flush=True)

    el = (time.monotonic() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"  {n_done} tasks | {n_ok} verified ({n_ok/max(n_done,1):.1%}) | "
          f"{n_repaired} rescued by repair")
    print(f"  {el:.1f} minutes | {out}")
    print("="*60)


if __name__ == "__main__":
    main()
