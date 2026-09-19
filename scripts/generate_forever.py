"""Run generation until the free tier is genuinely exhausted, then wait for it.

Rate limits are per model, so the usable daily budget is the sum across every
model that answers. This rotates through them, retires one on a 429, and waits
rather than exiting when all are cooling down — a quota that resets in fifteen
minutes is worth waiting for on a job with weeks to run.

Concurrent, because the work is not CPU-bound in one place: generation waits on
the network while rendering waits on a subprocess, so a sequential loop leaves
both the quota and the cores idle. Concurrency costs nothing in quality — each
task is independent and every result passes the same render gate.

Safe to stop at any time. Results are written and flushed as they complete, and
completed task keys are skipped on restart.
"""
from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

STOP = False


def _handle(signum, frame):
    global STOP
    STOP = True
    print("\n  stop requested — finishing in-flight tasks, then exiting", flush=True)


def main() -> None:
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic/stream.jsonl")
    ap.add_argument("--max-tasks", type=int, default=1000000)
    ap.add_argument("--max-hours", type=float, default=20.0)
    ap.add_argument("--variations", type=int, default=1)
    ap.add_argument("--repair-rounds", type=int, default=2)
    ap.add_argument("--cooldown", type=float, default=900.0)
    ap.add_argument("--workers", type=int, default=5)
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
    queue = queue[: a.max_tasks]
    print(f"queue: {len(queue)} pending ({len(done)} already done)")
    print(f"models: {len(GEMINI_ROTATION)} in rotation | workers: {a.workers}\n", flush=True)

    pool = ModelPool(list(GEMINI_ROTATION), cooldown_s=a.cooldown)
    harness = RenderHarness(python_bin="./.venv/bin/python",
                            cache_dir="data/frames", timeout=90)

    t_start = time.monotonic()
    counters = {"done": 0, "ok": 0, "repaired": 0, "skipped": 0, "errors": 0}
    lock = threading.Lock()
    sink = out.open("a")

    def expired() -> bool:
        return STOP or (time.monotonic() - t_start) / 3600 > a.max_hours

    def record(rec: dict) -> None:
        with lock:
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            counters["done"] += 1
            if rec.get("skipped"):
                counters["skipped"] += 1
            if rec.get("ok"):
                counters["ok"] += 1
                if rec.get("repair_rounds"):
                    counters["repaired"] += 1
            n = counters["done"]
            if n % 10 == 0:
                el = max((time.monotonic() - t_start) / 60, 0.01)
                scored = n - counters["skipped"]
                yld = counters["ok"] / max(scored, 1)
                print(f"  [{n:>5}] verified {counters['ok']:>5} "
                      f"yield {yld:5.1%}  {n/el:4.1f}/min  {el:6.1f}m  "
                      f"[{pool.status()}]", flush=True)

    def do_task(task):
        if expired():
            return
        model = pool.next_model()
        while model is None:
            if expired():
                return
            time.sleep(min(30.0, max(5.0, pool.seconds_until_any())))
            model = pool.next_model()

        teacher = Teacher(provider="gemini", model=model)
        try:
            g = generate_and_repair(teacher, harness, task.request,
                                    task.n_beats, task.length_hint,
                                    max_rounds=a.repair_rounds)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "503" in msg:
                pool.retire(model)
            with lock:
                counters["errors"] += 1
            return

        if g.code.strip().upper().startswith("SKIP") or len(g.code) < 120:
            record({"task_key": task.key, "kind": task.kind, "skipped": True})
            return

        row = CorpusRow.build(
            source=f"stream-{task.kind}", license="CC-BY-NC-SA-4.0",
            prompt=task.request if task.kind == "topic" else task.request[-1400:],
            code=g.code, index=0,
            tags=["synthetic", task.kind, f"domain:{task.meta.get('domain','?')}"])
        rec = row.to_dict()
        rec.update({"task_key": task.key, "kind": task.kind,
                    "ok": g.ok, "error_kind": g.error_kind,
                    "repair_rounds": g.rounds, "history": g.history,
                    "teacher_model": getattr(teacher, "last_model_used", model),
                    "video_s": g.duration_s,
                    # Failed attempts kept in full: paired with the final code
                    # these are (broken -> error -> fixed) triples, the data
                    # that teaches a model to read its own traceback.
                    "attempts": [{"code": s.code, "error_kind": s.error_kind,
                                  "stderr_tail": s.stderr_tail} for s in g.attempts],
                    **task.meta})
        record(rec)

    try:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futures = [ex.submit(do_task, t) for t in queue]
            for _ in as_completed(futures):
                if expired():
                    for f in futures:
                        f.cancel()
                    break
    finally:
        sink.close()

    el = (time.monotonic() - t_start) / 60
    scored = counters["done"] - counters["skipped"]
    print(f"\n{'='*64}")
    print(f"  {counters['done']} tasks | {counters['ok']} verified "
          f"({counters['ok']/max(scored,1):.1%} of {scored} scored)")
    print(f"  {counters['skipped']} skipped as non-visual | "
          f"{counters['repaired']} rescued by repair | {counters['errors']} API errors")
    print(f"  {el:.1f} min | {out}")
    print("="*64)


if __name__ == "__main__":
    main()
