#!/usr/bin/env python3
"""Wait for enough synthetic arcs, then build and push planner v2.

    ./.venv/bin/python scripts/train_planner_v2.py --min-arcs 400

Planner v1 trained 154 steps on 1,226 windows and its eval loss went
1.621 -> 1.592 -> 1.590: flattening, so another epoch on the same rows buys
little and more rows buy more. `synth_plans.py` is producing those.

This waits rather than being run by hand, because the GPU slot it needs is
occupied by the coder anyway and the arcs arrive over hours. It pushes when
**both** are true: enough arcs, and a free Kaggle session.

Synthetic arcs are weighted **below** the real ones. 146 real 3Blue1Brown
arcs are the thing being imitated; a teacher's imitation of them is worth
less, and mixing the two at equal weight would make the next planner number
say nothing about which taught it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
SYNTH = ROOT / "data" / "planner" / "plan_synth.jsonl"


def rows(path: Path) -> int:
    return sum(1 for l in path.open() if l.strip()) if path.exists() else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-arcs", type=int, default=400)
    ap.add_argument("--poll", type=int, default=600)
    ap.add_argument("--max-wait-hours", type=float, default=10.0)
    a = ap.parse_args()

    deadline = time.time() + a.max_wait_hours * 3600
    while rows(SYNTH) < a.min_arcs:
        if time.time() > deadline:
            print(f"only {rows(SYNTH)} arcs after {a.max_wait_hours}h; "
                  f"training on what there is", flush=True)
            break
        print(f"  {rows(SYNTH)}/{a.min_arcs} synthetic arcs", flush=True)
        time.sleep(a.poll)

    n = rows(SYNTH)
    print(f"building planner v2 with {n} synthetic arcs", flush=True)
    for cmd in (
        [str(PY), str(ROOT / "scripts" / "build_planner_windows.py"),
         "--src", str(SYNTH),
         "--out", str(ROOT / "data" / "planner" / "plan_synth_windows.jsonl")],
        [str(PY), str(ROOT / "scripts" / "build_coder_dataset.py"),
         "--which", "planner"],
    ):
        r = subprocess.run(cmd)
        if r.returncode != 0:
            return r.returncode

    # Push when a GPU session frees. Kaggle allows two and gives no cancel,
    # so waiting is the only option and a loop is cheaper than a person.
    for attempt in range(1, 121):
        out = subprocess.run(
            [str(PY), str(ROOT / "scripts" / "push_kernel.py"), "planner"],
            capture_output=True, text=True)
        if "successfully pushed" in out.stdout + out.stderr:
            print(f"planner v2 pushed on attempt {attempt}", flush=True)
            return 0
        print(f"  attempt {attempt}: no free GPU session", flush=True)
        time.sleep(600)
    print("never got a session", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
