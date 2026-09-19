"""Phase 2 — the render gate.

Execute every corpus row and keep what survives. This is the step that makes
the dataset worth more than scraped text: code that has never been run is a
guess about Manim's API, and six years of version drift means a lot of those
guesses are wrong.

Three properties matter more than speed here:

**Resumable.** Gating thousands of scenes takes hours on a fanless laptop that
will thermally throttle and that its owner needs for other things. Results
stream to disk as they complete and completed ids are skipped on restart, so
the job can be stopped and resumed freely.

**Parallel, but not greedy.** Renders are separate processes, so they scale
across cores — but saturating all ten makes the machine unusable and, on
passive cooling, throttles into being slower than using fewer.

**Honest about blame.** Environment failures are recorded separately and never
counted against a row. If TeX breaks mid-run, those rows are retried later
rather than silently discarded as bad data.
"""

from __future__ import annotations

import json
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from forge.harness import RenderHarness, ErrorKind
from forge.repair.lint import lint

_HARNESS: RenderHarness | None = None


def _init_worker(python_bin: str, cache_dir: str, timeout: int) -> None:
    """One harness per worker process, created once rather than per task."""
    global _HARNESS
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # let the parent own Ctrl-C
    _HARNESS = RenderHarness(python_bin=python_bin, cache_dir=cache_dir, timeout=timeout)


def _gate_one(payload: tuple[str, str, str | None]) -> dict:
    row_id, code, scene = payload
    assert _HARNESS is not None
    t = time.monotonic()

    # Lint before rendering. Roughly two thirds of ManimBench calls self.add
    # and never animates, which renders zero frames and writes no video. That
    # is valid Manim producing a still image, not broken code, so a trailing
    # self.wait() recovers the row. It stays tagged as static (n_play_calls is
    # 0 in the corpus) so the training mix can weight it down.
    code, rules = lint(code)

    r = _HARNESS.render(code, scene_class=scene, quality="low", frames=4)
    return {
        "id": row_id,
        "ok": r.ok,
        "lint": rules,
        "code": code if (r.ok and rules) else None,   # keep repaired source
        "error_kind": r.error_kind.value,
        "is_env_failure": r.is_environment_failure,
        "n_frames": len(r.frame_paths),
        "duration_s": r.duration_s,
        "elapsed_s": round(time.monotonic() - t, 2),
        "stderr_tail": r.feedback(8) if not r.ok else "",
    }


def load_done(path: Path) -> set[str]:
    """Ids already gated, so a resumed run does no work twice.

    Environment failures are deliberately *not* counted as done. A row that
    failed because TeX was missing or a worker crashed has told us nothing
    about the code, so it must get another chance on the next run rather than
    being written off as bad data.
    """
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("is_env_failure"):
                continue
            if "id" in rec:
                done.add(rec["id"])
    return done


def gate_corpus(corpus: Path, out: Path, workers: int = 6,
                timeout: int = 90, limit: int | None = None,
                python_bin: str = "./.venv/bin/python") -> dict:
    rows = [json.loads(l) for l in corpus.open()]
    if limit:
        rows = rows[:limit]

    done = load_done(out)
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} rows | {len(done)} already gated | {len(todo)} to do")
    if not todo:
        return summarise(out)

    out.parent.mkdir(parents=True, exist_ok=True)
    payloads = [(r["id"], r["code"], r.get("scene_class")) for r in todo]

    t0 = time.monotonic()
    n_ok = n_done = 0
    # Append mode and flush per line: a kill -9 loses at most one result.
    with out.open("a") as sink, ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker,
        initargs=(os.path.abspath(python_bin), "data/frames", timeout),
    ) as pool:
        futures = {pool.submit(_gate_one, p): p[0] for p in payloads}
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"id": futures[fut], "ok": False,
                       "error_kind": "worker_crash", "is_env_failure": True,
                       "stderr_tail": f"{type(e).__name__}: {e}"}
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            n_done += 1
            n_ok += rec["ok"]
            if n_done % 25 == 0 or n_done == len(todo):
                el = time.monotonic() - t0
                rate = n_done / el
                eta = (len(todo) - n_done) / rate / 60 if rate else 0
                print(f"  {n_done:>5}/{len(todo)}  pass {n_ok/n_done:5.1%}  "
                      f"{rate:4.1f}/s  eta {eta:5.1f} min", flush=True)

    return summarise(out)


def summarise(out: Path) -> dict:
    from collections import Counter
    recs = [json.loads(l) for l in out.open()]
    scored = [r for r in recs if not r.get("is_env_failure")]
    n_ok = sum(r["ok"] for r in scored)
    return {
        "total": len(recs),
        "scored": len(scored),
        "env_failures": len(recs) - len(scored),
        "passed": n_ok,
        "pass_rate": round(n_ok / len(scored), 4) if scored else 0.0,
        "failures": dict(Counter(r["error_kind"] for r in scored if not r["ok"]).most_common()),
    }
