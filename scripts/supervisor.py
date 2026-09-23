#!/usr/bin/env python3
"""Watch every job, restart what died, and write down what each has produced.

    ./.venv/bin/python scripts/supervisor.py --once     # one sweep
    ./.venv/bin/python scripts/supervisor.py            # every 30 minutes

This project has twice reported a resource as working when it was producing
nothing: four daemons of which two had empty queues, and a planner build that
sat dead for twenty minutes while its log said "running". So a job here is
described by **what it has produced**, not by whether a process exists. A
process that is alive and whose output has not grown in an hour is reported
as stalled, which is the state that actually costs a night.

Each job declares:

  alive    how to tell it is running
  produced a number that must go up -- rows on disk, files, a status
  restart  what to run if it is dead and unfinished
  done     when it needs no restarting

Restarts are throttled and counted. A job that will not stay up is reported
rather than relaunched forever; a restart loop is the busy-looking failure
this project has already built once.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
KAGGLE = ROOT / ".venv" / "bin" / "kaggle"
STATE = ROOT / "data" / "supervisor" / "state.json"
LOG = ROOT / "data" / "supervisor" / "supervisor.log"
MAX_RESTARTS = 3
STALL_SWEEPS = 2          # sweeps with no growth before calling it stalled


def say(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%SZ}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def running(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern],
                          capture_output=True).returncode == 0


def lines_in(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open() if line.strip())


def kernel_status(ref: str) -> str:
    r = subprocess.run([str(KAGGLE), "kernels", "status", ref],
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr)
    for word in ("complete", "error", "cancel", "running", "queued"):
        if word in out.lower():
            return word
    return "unknown"


@dataclass
class Job:
    name: str
    kind: str                      # "local" | "kaggle"
    produced: callable             # -> int
    alive: callable                # -> bool
    done: callable                 # -> bool
    restart: list[str] | None = None
    log: Path | None = None
    note: str = ""
    history: list[int] = field(default_factory=list)


def local(name: str, pattern: str, out: Path, cmd: list[str] | None,
          done_when=None, log: Path | None = None, note: str = "") -> Job:
    return Job(name=name, kind="local",
               produced=lambda: lines_in(out),
               alive=lambda: running(pattern),
               done=done_when or (lambda: False),
               restart=cmd, log=log, note=note)


def kaggle(name: str, ref: str, note: str = "") -> Job:
    return Job(name=name, kind="kaggle",
               produced=lambda: {"complete": 2, "running": 1, "queued": 1}
               .get(kernel_status(ref), 0),
               alive=lambda: kernel_status(ref) in ("running", "queued"),
               done=lambda: kernel_status(ref) == "complete",
               restart=None, note=note)


def jobs() -> list[Job]:
    d = ROOT / "data" / "planner"
    return [
        kaggle("coder-sft", "nimbou/manim-forge-coder-sft",
               "beat -> method, 9,504 rows"),
        kaggle("planner-sft", "nimbou/manim-forge-planner-sft",
               "windowed arcs, 1,226 rows"),
        local("synth-plans", "synth_plans.py", d / "plan_synth.jsonl",
              [str(PY), "-u", str(ROOT / "scripts" / "synth_plans.py")],
              log=ROOT / "data" / "logs" / "synth_plans.log",
              note="teacher-written arcs for corpus topics"),
    ]


def sweep(state: dict, restart: bool) -> dict:
    out = {}
    for job in jobs():
        prev = state.get(job.name, {})
        n = job.produced()
        hist = (prev.get("history") or [])[-4:] + [n]
        alive, finished = job.alive(), job.done()
        stalled = (alive and len(hist) > STALL_SWEEPS
                   and len(set(hist[-STALL_SWEEPS - 1:])) == 1)
        status = ("done" if finished else "stalled" if stalled
                  else "running" if alive else "down")
        restarts = prev.get("restarts", 0)

        if status == "down" and not finished and job.restart and restart:
            if restarts >= MAX_RESTARTS:
                status = "down (gave up)"
            else:
                logf = job.log or (ROOT / "data" / "logs" / f"{job.name}.log")
                logf.parent.mkdir(parents=True, exist_ok=True)
                with logf.open("a") as f:
                    subprocess.Popen(job.restart, stdout=f, stderr=f,
                                     start_new_session=True, cwd=ROOT)
                restarts += 1
                status = "restarted"
                say(f"  {job.name}: was down, relaunched ({restarts}/{MAX_RESTARTS})")

        out[job.name] = {"status": status, "produced": n, "history": hist,
                         "restarts": restarts, "kind": job.kind,
                         "note": job.note}
        say(f"  {job.name:14s} {status:15s} produced={n:<7} {job.note}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=int, default=1800)
    ap.add_argument("--no-restart", action="store_true")
    a = ap.parse_args()

    STATE.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    if STATE.exists():
        try:
            state = json.loads(STATE.read_text()).get("jobs", {})
        except json.JSONDecodeError:
            pass
    while True:
        say("=" * 62)
        state = sweep(state, restart=not a.no_restart)
        STATE.write_text(json.dumps(
            {"updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "jobs": state}, indent=2) + "\n")
        if a.once:
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    raise SystemExit(main())
