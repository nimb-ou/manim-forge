"""The always-on worker pool.

One process to leave running. It starts every job, watches what each one is
actually producing, restarts what has died, and keeps the machine busy when
the API quota runs out -- which it does every day, for hours.

Health is measured as progress, never as existence. A process sitting at 0%
CPU with a thread pool politely waiting on a rate limit is alive and useless,
and a supervisor that cannot tell the difference will report it healthy
forever. This one restarts a stalled job and merely notes a parked one, the
difference being whether the resource it needs is available.

    ./.venv/bin/python -u scripts/forge_run.py                 # everything
    ./.venv/bin/python -u scripts/forge_run.py --only cpu      # no API jobs
    ./.venv/bin/python -u scripts/forge_run.py --status        # one report
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge.orchestrator import Health, Job, Resource, api_is_live  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "data" / "logs"
PY = "./.venv/bin/python"

STOP = False


def _stop(*_):
    global STOP
    STOP = True
    print("\n  shutting down: stopping children", flush=True)


def roster() -> list[Job]:
    """Every job worth running, derived from the same backlog the doctor reads.

    Not a hand-maintained list. The two drifted before: the doctor knew
    twenty-five showcase renders were outstanding while the pool -- which I
    had emptied that morning -- was not running at all, and the gap survived
    an entire training run because nothing compared them.

    `forge.doctor.backlog()` is now the single definition of what work
    exists, what resource it needs, how to start it, and why it is parked if
    it is. A job appears here when it has outstanding units, a command, and
    no `blocked_by`.

    The pool is deliberately small right now -- one job. Corpus generation is
    parked until Phase 2 gives us a measure for whether a generated row is
    worth keeping, because more rows from the same teacher is the specific
    thing docs/PLAN.md says makes the model worse. A one-job pool that is
    honest beats a four-job pool where two are doing nothing, which is what
    this was yesterday.
    """
    from forge.doctor import backlog

    patience = {"cpu": 7200, "api": 1200}
    jobs = []
    for w in backlog():
        if w.blocked_by or not w.command or w.outstanding == 0:
            continue
        resource = Resource(w.resource) if w.resource in ("cpu", "api", "gpu") \
            else None
        if resource is None:
            continue                      # "claude" work: mine, not a process
        name = w.name.split()[0]
        jobs.append(Job(
            name=name,
            command=w.command.replace("./.venv/bin/python", PY),
            resource=resource,
            counter=ROOT / _COUNTER[name],
            log=LOGS / f"{name}.log",
            patience_s=patience.get(w.resource, 1800),
            max_restarts=40,
            parks_when_blocked=(resource is Resource.API),
        ))
    return jobs


#: Where each job's output accumulates, so progress is measured rather than
#: assumed. A job whose line count is not moving is stalled even if its
#: process is perfectly alive.
_COUNTER = {
    "showcase": "data/showcase/rendered.jsonl",
    "re-gate": "data/verified/regate.jsonl",
}


def child_env() -> dict:
    env = dict(os.environ)
    env["PATH"] = "/Library/TeX/texbin:" + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    return env


def line(job: Job, h: Health) -> str:
    n = job.count()
    age = int(time.time() - job.last_progress_t)
    mark = {Health.WORKING: "  ", Health.STARTING: "..",
            Health.PARKED: "zz", Health.STALLED: "!!",
            Health.DEAD: "XX", Health.DONE: "ok"}[h]
    return (f"  {mark} {job.name:<14} {job.resource.value:<4} "
            f"{h.value:<9} {n:>7} produced   {age:>5}s since progress"
            + (f"   restarts {job.restarts}" if job.restarts else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["api", "cpu", "gpu"], default=None)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--status", action="store_true",
                    help="print one report and exit; starts nothing")
    a = ap.parse_args()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    jobs = [j for j in roster()
            if a.only is None or j.resource.value == a.only]
    LOGS.mkdir(parents=True, exist_ok=True)

    if a.status:
        live = api_is_live()
        print(f"api {'LIVE' if live else 'exhausted'}\n")
        for j in jobs:
            print(f"  {j.name:<14} {j.resource.value:<4} "
                  f"{j.count():>7} produced")
        return

    env = child_env()
    print(f"forge_run: {len(jobs)} jobs, checking every {a.interval:.0f}s\n",
          flush=True)
    for j in jobs:
        j.start(ROOT, env)
        print(f"  started {j.name} (pid {j.proc.pid})", flush=True)

    api_live = api_is_live()
    last_probe = time.time()
    tick = 0
    try:
        while not STOP:
            time.sleep(a.interval)
            tick += 1

            # Probing costs a request against the quota being measured, so do
            # it sparingly -- and immediately whenever an API job looks stuck,
            # since that is the one moment the answer changes what we do.
            need_probe = time.time() - last_probe > 600
            states = {}
            for j in jobs:
                states[j.name] = j.health(api_live)
            if any(states[j.name] in (Health.STALLED, Health.PARKED)
                   and j.resource is Resource.API for j in jobs):
                need_probe = True
            if need_probe:
                api_live = api_is_live()
                last_probe = time.time()
                for j in jobs:
                    states[j.name] = j.health(api_live)

            print(f"[{datetime.now():%H:%M:%S}] api "
                  f"{'live' if api_live else 'exhausted'}", flush=True)
            for j in jobs:
                h = states[j.name]
                print(line(j, h), flush=True)

                if h in (Health.DEAD, Health.STALLED):
                    if j.restarts >= j.max_restarts:
                        print(f"     {j.name}: {j.restarts} restarts, "
                              f"leaving it down", flush=True)
                        continue
                    if h is Health.STALLED:
                        j.stop()
                    j.restarts += 1
                    j.start(ROOT, env)
                    print(f"     restarted {j.name} "
                          f"(#{j.restarts}, pid {j.proc.pid})", flush=True)
                elif h is Health.DONE:
                    # Finished cleanly. Start another pass only if the last
                    # one actually produced something -- a job that exits at
                    # once because its queue is empty will do so again, and
                    # respawning it on a timer is a busy loop wearing the
                    # costume of a worker.
                    produced = j.count() > j.last_count_at_start
                    if (j.resource is Resource.CPU and tick % 30 == 0
                            and produced):
                        j.restarts += 1
                        j.start(ROOT, env)
                        print(f"     {j.name} finished; started another pass",
                              flush=True)
                    elif not produced and not j.exhausted_logged:
                        j.exhausted_logged = True
                        print(f"     {j.name} finished with nothing to do; "
                              f"leaving it down", flush=True)
            print(flush=True)
    finally:
        for j in jobs:
            j.stop()
        print("  all children stopped", flush=True)


if __name__ == "__main__":
    main()
