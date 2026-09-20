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
    """Every job, with how to tell whether it is doing anything.

    The CPU jobs exist so that ten cores are not idle for the ten hours a day
    the Gemini quota is exhausted.

    regate is **not** here, and the reason is worth stating. It has now run
    the full 1,920 previously-failed rows against the current four lint
    rules: a rule fired on 122 of them and 8 rows were recovered. It is
    finished until a new rule lands. Left in the roster it found nothing to
    do, exited cleanly, and got restarted every thirty ticks forever -- work
    that looked like work. Run it by hand after adding a lint rule, which is
    the only moment it can produce anything:

        ./.venv/bin/python scripts/regate.py --workers 4
    """
    return [
        Job(name="code-synth",
            command=f"{PY} -u scripts/generate_forever.py --max-hours 24 "
                    f"--workers 5 --repair-rounds 4",
            resource=Resource.API,
            counter=ROOT / "data/synthetic/stream.jsonl",
            log=LOGS / "stream.log",
            patience_s=1200, parks_when_blocked=True),

        Job(name="prose-synth",
            command=f"{PY} -u scripts/generate_tasks.py --per-kind 600 "
                    f"--workers 3 --max-hours 24",
            resource=Resource.API,
            counter=ROOT / "data/synthetic/tasks.jsonl",
            log=LOGS / "tasks.log",
            patience_s=1200, parks_when_blocked=True),

        Job(name="showcase",
            command=f"{PY} -u scripts/render_showcase.py --quality high",
            resource=Resource.CPU,
            counter=ROOT / "data/showcase/rendered.jsonl",
            log=LOGS / "showcase.log",
            # One 1080p60 scene can legitimately take half an hour, and the
            # 3-D one takes longer, so patience here is measured in hours.
            patience_s=7200, max_restarts=40),

        # gold-verify is deliberately absent. It rendered all 44 scenes at
        # 480p15 while showcase rendered the same 44 at 1080p60 -- the same
        # work twice, six manim processes deep, for a load average of 95 on
        # ten cores. The higher-quality pass proves the scene renders and
        # produces the asset we actually want; the low-quality pass proved
        # only the first half. Run verify_gold by hand when iterating on one
        # scene, where its speed is the point.
    ]


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
