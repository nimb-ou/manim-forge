"""The two bugs that took the machine to a load average of 99.

Both were a cleanup path that named the right process group and then failed
to signal it. Both are tested the same way: run something that spawns a
grandchild, kill the parent, and check whether the grandchild is still
ticking a file forward a few seconds later.

See docs/POSTMORTEM.md, C1 and C2.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from forge.harness.render import _run_bounded
from forge.orchestrator import Job, Resource

TICKER = """
import time
for i in range(120):
    open({mark!r}, 'w').write(str(i))
    time.sleep(0.2)
"""


def _spawner(mark: str, term_handler: bool) -> str:
    """A process that launches a ticking grandchild, then sleeps past any test."""
    handler = ("import signal, os\n"
               "signal.signal(signal.SIGTERM, lambda *a: os._exit(0))\n"
               ) if term_handler else ""
    return (
        "import subprocess, sys, time\n"
        + handler +
        f"subprocess.Popen([sys.executable, '-c', {TICKER.format(mark=mark)!r}])\n"
        "time.sleep(120)\n"
    )


def _ticked(mark: str, window: float = 1.5) -> bool:
    """Is something still advancing the marker?"""
    before = Path(mark).read_text()
    time.sleep(window)
    return Path(mark).read_text() != before


@pytest.fixture
def mark(tmp_path: Path) -> str:
    return str(tmp_path / "tick")


def _wait_for(mark: str, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if Path(mark).exists():
            return
        time.sleep(0.1)
    raise AssertionError("grandchild never started")


def test_render_timeout_takes_grandchildren_with_it(mark, tmp_path):
    """subprocess.run(timeout=) kills one pid. Manim spawns ffmpeg and LaTeX.

    start_new_session=True puts the renderer in its own process group, which
    only helps if something then signals that group -- so on its own it makes
    orphaning *more* certain, not less.
    """
    script = tmp_path / "spawner.py"
    script.write_text(_spawner(mark, term_handler=False))

    started = time.monotonic()
    _out, _err, timed_out = _run_bounded(
        [sys.executable, str(script)], timeout=2, cwd=str(tmp_path),
        env=dict(os.environ))

    assert timed_out
    assert time.monotonic() - started < 15, "timeout did not return promptly"
    _wait_for(mark)
    assert not _ticked(mark), "grandchild outlived the timeout"


def test_job_stop_sweeps_a_pool_that_outlives_its_parent(mark, tmp_path):
    """The parent exits at once on SIGTERM; its workers do not.

    The old code escalated to SIGKILL only `if self.alive()` -- asking about
    the parent, which was already gone -- so the sweep never ran.
    """
    script = tmp_path / "daemon.py"
    script.write_text(_spawner(mark, term_handler=True))

    job = Job(name="t", resource=Resource.CPU,
              command=f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}")
    job.start(tmp_path, dict(os.environ))
    try:
        _wait_for(mark)
        assert _ticked(mark), "nothing was running to begin with"
        job.stop()
        assert not _ticked(mark), "pool worker survived Job.stop()"
    finally:
        job.stop()


def test_stop_is_safe_on_a_job_that_already_exited(tmp_path):
    """Signalling an empty group raises ProcessLookupError; stop() absorbs it."""
    job = Job(name="t", resource=Resource.CPU, command=f"{shlex.quote(sys.executable)} -c pass")
    job.start(tmp_path, dict(os.environ))
    job.proc.wait(timeout=10)
    job.stop()          # must not raise
    job.stop()          # and must be idempotent


def test_stop_on_a_job_that_never_started():
    Job(name="t", resource=Resource.CPU, command="true").stop()
