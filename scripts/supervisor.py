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
import signal
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
# 12, not 3. When restarts were a guess at an unknown fault, three was a
# sensible ceiling. Now a restart is a known recovery from a known fault --
# a teacher call that hangs despite a client timeout -- and capping at three
# means the job stays dead for the rest of the night after three bad calls.
# The ceiling still exists so a job that cannot start at all is reported
# rather than relaunched forever.
MAX_RESTARTS = 12
STALL_SWEEPS = 2          # sweeps with no growth before calling it stalled


def say(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%SZ}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


# Every subprocess call here gets a timeout. The supervisor swept at 06:41
# and then not again until 07:25 -- 44 minutes blocked inside a `kaggle
# kernels status` call that never returned -- and during that window the job
# it was watching stalled and was not noticed. A watchdog that blocks on the
# same kind of call it is watching for is not a watchdog.
TIMEOUT = 45


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=TIMEOUT)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    except Exception as exc:                                  # noqa: BLE001
        return 125, f"{type(exc).__name__}: {exc}"


def running(pattern: str) -> bool:
    return _run(["pgrep", "-f", pattern])[0] == 0


def lines_in(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open() if line.strip())


_STATUS: dict[str, str] = {}


def kernel_status(ref: str, fresh: bool = False) -> str:
    """One API call per kernel per sweep, not three.

    `kaggle()` builds produced, alive and done as separate lambdas and each
    one called this, so a sweep made six Kaggle calls for two kernels -- up
    to 270 seconds of blocking at the 45-second timeout, against a
    five-minute interval. That is most of why sweeps went missing.
    """
    if not fresh and ref in _STATUS:
        return _STATUS[ref]
    code, out = _run([str(KAGGLE), "kernels", "status", ref])
    # A timed-out status is "unreachable", not failed: a slow Kaggle API is
    # not a broken kernel, and reporting it as one restarts healthy things.
    verdict = "unknown"
    if code == 124:
        verdict = "unreachable"
    else:
        for word in ("complete", "error", "cancel", "running", "queued"):
            if word in out.lower():
                verdict = word
                break
    _STATUS[ref] = verdict
    return verdict


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
    # Some jobs are supposed to sit at zero. planner-v2 waits for 400 arcs
    # before it does anything, so flat output is it working correctly, and
    # reporting that as stalled is the third always-wrong alarm this file has
    # produced. Flat output only means something for a job that should be
    # producing *now*.
    stallable: bool = True
    out: Path | None = None
    history: list[int] = field(default_factory=list)


def local(name: str, pattern: str, out: Path, cmd: list[str] | None,
          done_when=None, log: Path | None = None, note: str = "",
          stallable: bool = True) -> Job:
    # A job whose output is one artefact written at the end has no growing
    # count, so "stalled" would be wrong for it the whole way through. Those
    # report 1 when the artefact exists and 0 before, and are only ever
    # running or done.
    return Job(name=name, kind="local",
               produced=lambda: (lines_in(out) if out.suffix == ".jsonl"
                                 else int(out.exists())),
               alive=lambda: running(pattern),
               done=done_when or (lambda: False),
               restart=cmd, log=log, note=note, stallable=stallable,
               out=out)


def kaggle(name: str, ref: str, note: str = "") -> Job:
    return Job(name=name, kind="kaggle",
               produced=lambda: {"complete": 2, "running": 1, "queued": 1}
               .get(kernel_status(ref), 0),
               alive=lambda: kernel_status(ref) in ("running", "queued",
                                                     "unreachable"),
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
              [str(PY), "-u", str(ROOT / "scripts" / "synth_plans.py"),
               "--provider", "mistral", "--model", "mistral-medium-latest"],
              log=ROOT / "data" / "logs" / "synth_plans.log",
              note="teacher-written arcs for corpus topics"),
        local("planner-v2", "train_planner_v2.py",
              d / "plan_synth_windows.jsonl",
              [str(PY), "-u", str(ROOT / "scripts" / "train_planner_v2.py"),
               "--min-arcs", "250"],
              log=ROOT / "data" / "logs" / "planner_v2.log",
              note="waits for arcs, then pushes planner v2", stallable=False),
        local("collect-coder2", "collect_adapter.*coder2",
              ROOT / "adapters" / "mlx-coder2" / "adapters.safetensors",
              [str(PY), "-u", str(ROOT / "scripts" / "collect_adapter.py"),
               "--kernel", "nimbou/manim-forge-coder-sft",
               "--peft", str(ROOT / "adapters" / "kaggle-coder2"),
               "--mlx", str(ROOT / "adapters" / "mlx-coder2")],
              log=ROOT / "data" / "logs" / "collect_coder.log",
              note="waits, downloads, converts, verifies the coder adapter",
              stallable=False),
        # plan-lengths is deliberately not listed. It was stopped to free the
        # CPU for the two-adapter run, and a supervisor that relists every
        # job ever started turns "down" into permanent noise.
    ]


def sweep(state: dict, restart: bool, stale_minutes: float = 25.0) -> dict:
    out = {}
    for job in jobs():
        prev = state.get(job.name, {})
        n = job.produced()
        hist = (prev.get("history") or [])[-4:] + [n]
        alive, finished = job.alive(), job.done()
        # Only jobs that grow a count can stall. A job that writes one
        # artefact at the end shows no growth by design, and calling that
        # stalled would be an alarm that is always wrong -- the kind you
        # learn to ignore, which is worse than no alarm.
        stalled = (alive and job.stallable and job.restart is not None
                   and len(hist) > STALL_SWEEPS
                   and len(set(hist[-STALL_SWEEPS - 1:])) == 1)
        # Wall clock, not sweep count. The history above showed a healthy
        # rising line through the stall, because the supervisor was itself
        # blocked and took no samples during it. The file's mtime does not
        # depend on the supervisor having been awake.
        # A restart does not touch the output file, so mtime stays old until
        # the new process lands its first row -- and a job that takes two
        # minutes an item would be killed again on the next sweep, forever.
        # That restart loop is the busy-looking failure this project has
        # already built once, so the clock starts at the restart.
        since_restart = (time.time() - prev.get("restarted_at", 0)) / 60
        if alive and job.stallable and job.out is not None \
                and job.out.exists() and since_restart > stale_minutes:
            idle = (time.time() - job.out.stat().st_mtime) / 60
            if idle > stale_minutes:
                stalled = True
                say(f"  {job.name}: no output for {idle:.0f} minutes")
        status = ("done" if finished else "stalled" if stalled
                  else "running" if alive else "down")
        restarts = prev.get("restarts", 0)
        restarted_at = prev.get("restarted_at", 0)

        if status in ("down", "stalled") and not finished \
                and job.restart and restart:
            if status == "stalled":
                _run(["pkill", "-f", job.restart[-1]])
            if restarts >= MAX_RESTARTS:
                status = "down (gave up)"
            else:
                logf = job.log or (ROOT / "data" / "logs" / f"{job.name}.log")
                logf.parent.mkdir(parents=True, exist_ok=True)
                with logf.open("a") as f:
                    subprocess.Popen(job.restart, stdout=f, stderr=f,
                                     start_new_session=True, cwd=ROOT)
                restarts += 1
                restarted_at = time.time()
                status = "restarted"
                say(f"  {job.name}: was down, relaunched ({restarts}/{MAX_RESTARTS})")

        out[job.name] = {"status": status, "produced": n, "history": hist,
                         "restarts": restarts, "restarted_at": restarted_at,
                         "kind": job.kind, "note": job.note}
        say(f"  {job.name:14s} {status:15s} produced={n:<7} {job.note}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=int, default=300,
                    help="seconds between sweeps. 300, not 1800: state.json "
                         "is what other watchers read, and a half-hour-old "
                         "snapshot reported 21 arcs when 136 were on disk. A "
                         "sweep is a few subprocess calls; staleness costs "
                         "more than the sweep does.")
    ap.add_argument("--no-restart", action="store_true")
    ap.add_argument("--retire-after", type=int, default=12,
                    help="exit cleanly after this many sweeps so launchd "
                         "hands over a fresh process. A loop that runs "
                         "forever is a loop that can wedge forever, and the "
                         "sweeps are cheap now (4.2s), so retiring hourly "
                         "costs nothing and bounds how long a wedge lasts.")
    ap.add_argument("--stale-minutes", type=float, default=25.0,
                    help="a job whose output file has not been written in "
                         "this long is stalled, regardless of what its row "
                         "count history says. The count history only sees "
                         "the sweeps that happened, and the sweeps stopped.")
    a = ap.parse_args()

    # A sweep that has not finished in two minutes is wedged, and the whole
    # point of this file is to notice wedged things. It ran as one
    # long-lived process sleeping between sweeps, and twice that process
    # stopped sweeping while staying alive -- once blocked in a Kaggle call,
    # once for no reason I could find, missing twelve consecutive sweeps at
    # zero CPU.
    #
    # So each sweep is now its own process, launched by launchd on an
    # interval, and it kills itself if it takes too long. A watchdog whose
    # own liveness has to be watched is not finished.
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(
                      TimeoutError("sweep took longer than 120s")))
    signal.alarm(120)

    STATE.parent.mkdir(parents=True, exist_ok=True)
    swept = 0
    state = {}
    if STATE.exists():
        try:
            state = json.loads(STATE.read_text()).get("jobs", {})
        except json.JSONDecodeError:
            pass
    while True:
        _STATUS.clear()          # one fresh reading per sweep
        say("=" * 62)
        state = sweep(state, restart=not a.no_restart,
                      stale_minutes=a.stale_minutes)
        STATE.write_text(json.dumps(
            {"updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "jobs": state}, indent=2) + "\n")
        signal.alarm(0)
        if a.once:
            return 0
        swept += 1
        if swept >= a.retire_after:
            say(f"retiring after {swept} sweeps; launchd starts the next")
            return 0
        # Slept in short steps, not one long one. A single time.sleep(300)
        # did not return for 38 minutes here -- under launchd and under
        # nohup, at nice 0 and at ProcessType Interactive, on a machine at
        # load 1.95. Whatever defers a five-minute timer on this system does
        # not defer a fifteen-second one, and the loop below reaches the same
        # wall-clock deadline by asking repeatedly instead of once.
        deadline = time.time() + a.every
        while time.time() < deadline:
            time.sleep(min(15, max(1, deadline - time.time())))
        signal.alarm(120)


if __name__ == "__main__":
    raise SystemExit(main())
