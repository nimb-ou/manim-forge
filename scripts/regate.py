"""Re-gate rows that failed, now that the lint has improved.

The gate ran before the stdlib-import rule existed, so 1,920 failures have
never been tested against it. Re-running them costs CPU and no API quota at
all — which is exactly what is free while the teacher's models sit in
cooldown.

Only previously-failed rows are retried. Rows that already passed are left
alone; re-rendering them would spend hours confirming what is already known.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_H = None


def _init(python_bin: str, cache: str, timeout: int) -> None:
    global _H
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from forge.harness import RenderHarness
    _H = RenderHarness(python_bin=python_bin, cache_dir=cache, timeout=timeout)


def _one(payload):
    rid, code, scene = payload
    from forge.repair.lint import RULES, lint
    fixed, rules = lint(code)
    if not rules:
        # Record which rules were *available*, not just which fired. A row
        # that no rule touched is not finished with -- it is finished with
        # under this rule set, and adding a rule must bring it back.
        return {"id": rid, "retried": False,
                "ruleset": _ruleset_id(RULES)}
    # use_cache=False: the cached verdict is for the unlinted source.
    r = _H.render(fixed, scene_class=scene, quality="low", frames=4, use_cache=False)
    from forge.repair.lint import RULES as _R
    return {"id": rid, "retried": True, "ok": r.ok,
            "error_kind": r.error_kind.value, "lint": rules,
            "ruleset": _ruleset_id(_R),
            "code": fixed if r.ok else None,
            "duration_s": r.duration_s, "n_frames": len(r.frame_paths)}


def _ruleset_id(rules) -> str:
    """A short stable name for the current lint rule set.

    Stored on every record so a later run can tell whether a row was judged
    under the same rules. 1,859 rows here were recorded as 'no rule fired'
    when only three rules existed; the percent-escape rule landed afterwards
    and none of them had ever been tested against it, while the resume set
    counted every one as done.
    """
    return ",".join(sorted(name for name, _, _ in rules))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/normalized/corpus.jsonl")
    ap.add_argument("--gate", default="data/verified/gate.jsonl")
    ap.add_argument("--out", default="data/verified/regate.jsonl")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=75)
    a = ap.parse_args()

    rows = {json.loads(l)["id"]: json.loads(l) for l in Path(a.corpus).open()}
    failed = []
    for line in Path(a.gate).open():
        v = json.loads(line)
        if not v.get("ok") and not v.get("is_env_failure"):
            r = rows.get(v["id"])
            if r:
                failed.append((r["id"], r["code"], r.get("scene_class")))

    from forge.repair.lint import RULES
    current = _ruleset_id(RULES)

    out = Path(a.out)
    done, stale = set(), 0
    if out.exists():
        for l in out.open():
            try:
                rec = json.loads(l)
            except json.JSONDecodeError:
                continue
            if rec.get("ruleset") == current:
                done.add(rec["id"])
            else:
                stale += 1
    todo = [f for f in failed if f[0] not in done]
    print(f"{len(failed)} previously failed | {len(done)} settled under the "
          f"current rules | {stale} judged under older rules | "
          f"{len(todo)} to retry", flush=True)
    print(f"ruleset: {current}", flush=True)

    n = ok = retried = 0
    t0 = time.monotonic()
    with out.open("a") as sink, ProcessPoolExecutor(
            max_workers=a.workers, initializer=_init,
            initargs=(os.path.abspath("./.venv/bin/python"), "data/frames", a.timeout)) as ex:
        futs = {ex.submit(_one, p): p[0] for p in todo}
        for f in as_completed(futs):
            try:
                rec = f.result()
            except Exception as e:
                rec = {"id": futs[f], "retried": False, "error": str(e)[:80]}
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            n += 1
            retried += bool(rec.get("retried"))
            ok += bool(rec.get("ok"))
            if n % 100 == 0:
                el = (time.monotonic() - t0) / 60
                print(f"  {n}/{len(todo)}  lint applied to {retried}  recovered {ok}  "
                      f"{el:.1f}m", flush=True)

    print(f"\n{'='*58}\n  {retried} rows had lint applied, {ok} now render")
    print(f"  corpus grows by {ok} at zero API cost\n{'='*58}")


if __name__ == "__main__":
    main()
