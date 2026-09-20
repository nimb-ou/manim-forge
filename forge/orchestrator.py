"""Job definitions and health checks for the always-on worker pool.

The problem this solves is specific and has bitten this project twice. A
supervisor that asks "is the process alive?" reports a stalled job as healthy
forever -- once for forty minutes with identical row counts, and once for an
hour while every Gemini model returned 429 and the CPU sat at 14% of one core
out of ten.

So health here is **progress**, not existence. A job declares how to measure
what it has produced, and a job that has produced nothing for its patience
window is restarted or parked, whichever is right for why it stopped.

The second idea is that jobs declare which resource they consume. API work
parks when the quota is gone; CPU work must then expand to fill the machine,
because there is always render and re-gate work outstanding and no reason for
ten cores to idle while a rate limit clears.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Resource(str, Enum):
    API = "api"          # bounded by the Gemini daily quota
    CPU = "cpu"          # bounded by cores; always something to do
    GPU = "gpu"          # bounded by the one Apple GPU; strictly one at a time


class Health(str, Enum):
    STARTING = "starting"
    WORKING = "working"
    STALLED = "stalled"          # alive but producing nothing
    PARKED = "parked"            # alive, producing nothing, and rightly so
    DEAD = "dead"
    DONE = "done"                # finished its work on purpose


@dataclass
class Job:
    name: str
    command: str
    resource: Resource
    #: File whose line count measures output. A job with no counter is judged
    #: by its log's modification time instead.
    counter: Path | None = None
    log: Path | None = None
    #: Seconds of no progress before the job is considered stalled. Generous
    #: for anything that renders, since one scene can legitimately take half
    #: an hour.
    patience_s: float = 900.0
    #: Restart a dead job at most this often, to avoid a crash loop.
    max_restarts: int = 20
    #: When true, no progress is treated as PARKED rather than STALLED if the
    #: resource it needs is unavailable.
    parks_when_blocked: bool = False

    proc: subprocess.Popen | None = field(default=None, repr=False)
    restarts: int = 0
    #: count at the moment of the last start, so "did this pass produce
    #: anything" is answerable without trusting the job's own reporting.
    last_count_at_start: int = 0
    exhausted_logged: bool = False
    last_count: int = -1
    last_progress_t: float = 0.0
    started_t: float = 0.0

    # -- measurement ---------------------------------------------------------

    def count(self) -> int:
        """How much this job has produced, by its own measure."""
        if self.counter is None:
            if self.log and self.log.exists():
                return int(self.log.stat().st_mtime)
            return 0
        if not self.counter.exists():
            return 0
        try:
            with self.counter.open("rb") as fh:
                return sum(1 for _ in fh)
        except OSError:
            return self.last_count

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def health(self, api_live: bool) -> Health:
        now = time.time()
        if not self.alive():
            if self.proc is not None and self.proc.returncode == 0:
                return Health.DONE
            return Health.DEAD
        if now - self.started_t < 90:
            return Health.STARTING
        n = self.count()
        if n > self.last_count:
            self.last_count = n
            self.last_progress_t = now
            return Health.WORKING
        if now - self.last_progress_t < self.patience_s:
            return Health.WORKING
        if self.parks_when_blocked and not api_live:
            return Health.PARKED
        return Health.STALLED

    # -- lifecycle -----------------------------------------------------------

    def start(self, root: Path, env: dict) -> None:
        if self.log:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            handle = self.log.open("a")
        else:
            handle = subprocess.DEVNULL
        self.proc = subprocess.Popen(
            shlex.split(self.command), cwd=root, env=env,
            stdout=handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
        self.started_t = self.last_progress_t = time.time()
        self.last_count = self.last_count_at_start = self.count()
        self.exhausted_logged = False

    def stop(self, grace: float = 4.0) -> None:
        """Terminate the job's whole process group, then kill it.

        Both signals go out unconditionally. The earlier version escalated to
        SIGKILL only `if self.alive()` -- that is, only if the *supervised
        process itself* was still running. A generation daemon dies promptly
        on SIGTERM while its worker pool and the manim renderers those
        workers launched do not, so the check said "already gone", the
        SIGKILL was skipped, and the group was left running with no handle
        on it. Twenty-four orphaned workers accumulated that way across a
        few restarts.

        Signalling a group that has already exited raises ProcessLookupError
        and does nothing else, so there is no cost to not guessing.
        """
        if self.proc is None:
            return
        try:
            pgid = os.getpgid(self.proc.pid)
        except (ProcessLookupError, PermissionError):
            self.proc = None
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError):
                break
            deadline = time.time() + grace
            while time.time() < deadline and self.alive():
                time.sleep(0.2)
        try:
            self.proc.wait(timeout=1)
        except Exception:
            pass
        self.proc = None


def api_is_live(timeout: float = 20.0) -> bool:
    """One cheap probe: is any model in the rotation answering?

    Cheap on purpose. Asking each of ten models costs ten requests against the
    very quota being measured, so this asks one and trusts the rotation to
    find the rest.
    """
    try:
        from forge.synth.teacher import GEMINI_REST, _load_dotenv
        import urllib.error
        import urllib.request
        _load_dotenv()
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            return False
        body = json.dumps({
            "contents": [{"parts": [{"text": "ok"}]}],
            "generationConfig": {"maxOutputTokens": 512},
        }).encode()
        req = urllib.request.Request(
            f"{GEMINI_REST}/models/gemini-3-flash-preview:generateContent",
            data=body, headers={"Content-Type": "application/json",
                                "x-goog-api-key": key})
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except urllib.error.HTTPError as e:
        return e.code not in (429, 403)
    except Exception:
        return False
