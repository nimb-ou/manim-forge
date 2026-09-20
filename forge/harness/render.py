"""Sandboxed Manim render harness.

The single most reused component in Manim Forge. Three callers depend on it:

* **Phase 2 — the render gate.** Every corpus row is executed here. Rows that
  fail are either repaired or dropped. This is what makes the dataset worth
  more than scraped text.
* **Phase 6 — evaluation.** Render Success Rate is literally the pass rate of
  this function over a held-out set.
* **Phase 7 — the product.** The generate/render/repair loop calls it on every
  model output, and feeds ``RenderResult.feedback()`` straight back to the model.

Design rules, in order of importance:

1. **Never exec candidate code in this process.** It is untrusted model output
   and half of it is broken by construction. Everything runs in a subprocess
   that can be killed.
2. **Always bounded.** Wall-clock timeout, killed as a process group so a
   wedged ffmpeg child cannot outlive its parent.
3. **Cacheable.** Renders are the expensive step and the corpus gets re-gated
   often. Results key off a hash of (code, quality), so re-gating is nearly free.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .errors import ErrorKind, classify, tail, is_environment_failure

# 480p15 for gating: fast enough to run a corpus through, detailed enough that
# frame comparison still means something. Final exports use "high".
QUALITY_FLAGS = {
    "low":    "-ql",   # 480p15
    "medium": "-qm",   # 720p30
    "high":   "-qh",   # 1080p60
}

DEFAULT_TIMEOUT = 90          # seconds; a gating scene that exceeds this is too complex
DEFAULT_FRAME_COUNT = 8       # sampled evenly across the video for visual comparison




def _run_bounded(cmd: list[str], timeout: int, cwd, env: dict
                 ) -> tuple[str, str, bool]:
    """Run a command, and on timeout kill the whole process group.

    `subprocess.run(timeout=...)` kills only the process it launched. Manim
    spawns ffmpeg and LaTeX underneath it, so a timeout there leaves those
    grandchildren running with nobody holding a handle to them -- which is
    how a laptop with ten cores reaches a load average of 99 while the
    supervisor reports every job stopped.

    `start_new_session=True` puts the renderer in its own process group, and
    that is only useful if something signals the group. This does.
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        stdin=subprocess.DEVNULL, cwd=cwd, env=env, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        return out or "", err or "", False
    except subprocess.TimeoutExpired:
        _killpg(proc)
        out, err = proc.communicate()
        return out or "", err or "", True


def _killpg(proc: subprocess.Popen, grace: float = 2.0) -> None:
    """SIGTERM the group, then SIGKILL it. Unconditionally, both times.

    Escalating only when the *direct child* is still alive is the trap: the
    parent dies promptly on SIGTERM and its children do not, so the check
    says "already gone" and the SIGKILL that would have swept them is never
    sent. Signalling an empty group raises ProcessLookupError and costs
    nothing, so there is no reason to guess.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=grace)
            if sig is signal.SIGTERM:
                # The leader is gone; sweep the group regardless.
                continue
            return
        except subprocess.TimeoutExpired:
            continue


@dataclass
class RenderResult:
    ok: bool
    code_hash: str
    scene_class: str | None = None
    error_kind: ErrorKind = ErrorKind.NONE
    stderr: str = ""
    stdout: str = ""
    video_path: str | None = None
    frame_paths: list[str] = field(default_factory=list)
    duration_s: float | None = None      # length of the rendered animation
    elapsed_s: float = 0.0               # wall-clock cost of rendering it
    from_cache: bool = False

    @property
    def is_environment_failure(self) -> bool:
        """Failure caused by this machine rather than the code under test."""
        return is_environment_failure(self.error_kind)

    def feedback(self, lines: int = 12) -> str:
        """The repair prompt fragment handed back to the model on failure."""
        if self.ok:
            return ""
        return f"[{self.error_kind.value}]\n{tail(self.stderr, lines)}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["error_kind"] = self.error_kind.value
        return d


def code_hash(code: str, quality: str = "low", frames: int = DEFAULT_FRAME_COUNT) -> str:
    """Stable identity for a render. Whitespace-sensitive on purpose: a
    reformatted scene is the same animation, but proving that costs more than
    re-rendering it."""
    payload = f"{quality}|{frames}|{code}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def find_scene_classes(code: str) -> list[str]:
    """Names of Scene subclasses, via AST — never by executing the code.

    Matches any class whose bases mention Scene (``Scene``, ``ThreeDScene``,
    ``MovingCameraScene``, ``manim.Scene`` …), which covers the real corpus
    without maintaining a list of every Scene subclass Manim ships.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            name = ""
            if isinstance(base, ast.Name):
                name = base.id
            elif isinstance(base, ast.Attribute):
                name = base.attr
            if "Scene" in name:
                found.append(node.name)
                break
    return found


def _probe_duration(video: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None
    except (subprocess.SubprocessError, ValueError):
        return None


def _extract_frames(video: Path, out_dir: Path, count: int) -> list[Path]:
    """Sample ``count`` frames at evenly spaced fractions of the runtime.

    Fractions rather than fixed timestamps, so a 3-second scene and a
    30-second scene are compared at the same points in their arc. This is what
    makes SSIM/CLIP comparison meaningful in Phase 6.
    """
    duration = _probe_duration(video)
    if not duration or duration <= 0:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(count):
        frac = (i + 0.5) / count
        ts = duration * frac
        dest = out_dir / f"f{i:02d}.jpg"
        try:
            r = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{ts:.3f}",
                 "-i", str(video), "-frames:v", "1", "-q:v", "4", str(dest)],
                capture_output=True, timeout=30,
            )
            if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
                paths.append(dest)
        except subprocess.SubprocessError:
            continue
    return paths


class RenderHarness:
    """Executes Manim code in isolation and reports what happened.

    ``cache_dir`` stores one JSON record plus sampled frames per unique render,
    keyed by :func:`code_hash`. ``store_video`` is off by default: across a
    15k-row corpus the mp4s dwarf everything else on disk, while the sampled
    frames carry the signal we actually compare.
    """

    def __init__(
        self,
        workdir: Path | str | None = None,
        cache_dir: Path | str | None = None,
        python_bin: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        store_video: bool = False,
    ) -> None:
        self.workdir = Path(workdir) if workdir else Path(tempfile.gettempdir()) / "manim-forge"
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/frames")
        # Resolved eagerly: renders run with cwd set to a temp directory, so a
        # relative interpreter path ("./.venv/bin/python") would not survive.
        raw_bin = python_bin or os.environ.get("FORGE_PYTHON") or "python3"
        # abspath, never resolve(): a venv interpreter is a symlink to the base
        # interpreter, and following it would run outside the venv — where manim
        # is not installed.
        self.python_bin = os.path.abspath(raw_bin) if os.sep in str(raw_bin) else raw_bin
        self.timeout = timeout
        self.store_video = store_video
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _child_env() -> dict[str, str]:
        """Environment for the render subprocess.

        TeX is added explicitly: a GUI-launched app does not inherit the login
        shell's PATH, so without this every MathTex scene would fail with
        `latex_missing` on a machine where LaTeX is perfectly well installed.
        """
        env = {**os.environ, "PYTHONWARNINGS": "ignore"}
        texbin = "/Library/TeX/texbin"
        if Path(texbin).is_dir() and texbin not in env.get("PATH", ""):
            env["PATH"] = f"{texbin}:{env.get('PATH', '')}"
        return env

    # -- cache ---------------------------------------------------------------

    def _cache_slot(self, h: str) -> Path:
        # Two-level fan-out: a flat directory of 15k entries is painful on APFS.
        return self.cache_dir / h[:2] / h

    def _load_cached(self, h: str) -> RenderResult | None:
        record = self._cache_slot(h) / "result.json"
        if not record.exists():
            return None
        try:
            data = json.loads(record.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        data["error_kind"] = ErrorKind(data.get("error_kind", "none"))
        data["from_cache"] = True
        return RenderResult(**data)

    def _save_cached(self, result: RenderResult) -> None:
        slot = self._cache_slot(result.code_hash)
        slot.mkdir(parents=True, exist_ok=True)
        try:
            (slot / "result.json").write_text(json.dumps(result.to_dict(), indent=2))
        except OSError:
            pass

    # -- render --------------------------------------------------------------

    def render(
        self,
        code: str,
        scene_class: str | None = None,
        quality: str = "low",
        frames: int = DEFAULT_FRAME_COUNT,
        use_cache: bool = True,
    ) -> RenderResult:
        h = code_hash(code, quality, frames)

        if use_cache:
            cached = self._load_cached(h)
            # Environment failures are never cached as verdicts about the code:
            # install TeX and the same row must get a fresh chance.
            if cached is not None and not cached.is_environment_failure:
                return cached

        started = time.monotonic()

        if scene_class is None:
            # Distinguish "never parsed" from "parsed but defines no Scene" —
            # the repair loop handles these very differently.
            try:
                ast.parse(code)
            except SyntaxError as exc:
                result = RenderResult(
                    ok=False, code_hash=h, error_kind=ErrorKind.SYNTAX,
                    stderr=f"SyntaxError: {exc.msg} (line {exc.lineno})",
                    elapsed_s=time.monotonic() - started,
                )
                self._save_cached(result)
                return result

            candidates = find_scene_classes(code)
            if not candidates:
                result = RenderResult(
                    ok=False, code_hash=h, error_kind=ErrorKind.NO_SCENE,
                    stderr="No Scene subclass found in the submitted code.",
                    elapsed_s=time.monotonic() - started,
                )
                self._save_cached(result)
                return result
            # Last defined scene: files that build up helpers then finish with
            # the real scene are the common shape in this corpus.
            scene_class = candidates[-1]

        run_dir = Path(tempfile.mkdtemp(prefix=f"r-{h}-", dir=self.workdir))
        try:
            return self._run(code, scene_class, quality, frames, h, run_dir, started)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def _run(self, code, scene_class, quality, frames, h, run_dir, started) -> RenderResult:
        scene_file = run_dir / "scene.py"
        scene_file.write_text(code)
        media_dir = run_dir / "media"

        cmd = [
            self.python_bin, "-m", "manim",
            QUALITY_FLAGS.get(quality, "-ql"),
            "--disable_caching",
            "--media_dir", str(media_dir),
            str(scene_file), scene_class,
        ]

        timed_out = False
        stdout = stderr = ""
        try:
            stdout, stderr, timed_out = _run_bounded(
                cmd, timeout=self.timeout, cwd=run_dir, env=self._child_env())
        except OSError as exc:
            # The renderer never started: a machine problem, not a code verdict.
            elapsed = time.monotonic() - started
            return RenderResult(
                ok=False, code_hash=h, scene_class=scene_class,
                error_kind=ErrorKind.LAUNCH,
                stderr=f"Failed to launch renderer ({self.python_bin}): {exc}",
                elapsed_s=elapsed,
            )

        videos = sorted(media_dir.rglob("*.mp4")) if media_dir.exists() else []
        # partial_movie_files are per-animation fragments, not the finished scene.
        videos = [v for v in videos if "partial_movie_files" not in v.parts]

        elapsed = time.monotonic() - started

        if timed_out or not videos:
            kind = classify(stderr, timed_out=timed_out)
            if kind is ErrorKind.NONE and not videos:
                kind = ErrorKind.EMPTY_RENDER
            result = RenderResult(
                ok=False, code_hash=h, scene_class=scene_class, error_kind=kind,
                stderr=stderr[-8000:], stdout=stdout[-2000:], elapsed_s=elapsed,
            )
            if not result.is_environment_failure:
                self._save_cached(result)
            return result

        video = max(videos, key=lambda p: p.stat().st_size)
        slot = self._cache_slot(h)
        frame_paths = _extract_frames(video, slot / "frames", frames)

        stored_video: str | None = None
        if self.store_video:
            slot.mkdir(parents=True, exist_ok=True)
            dest = slot / "scene.mp4"
            shutil.copy2(video, dest)
            stored_video = str(dest)

        result = RenderResult(
            ok=True, code_hash=h, scene_class=scene_class, error_kind=ErrorKind.NONE,
            stdout=stdout[-2000:], video_path=stored_video,
            frame_paths=[str(p) for p in frame_paths],
            duration_s=_probe_duration(video), elapsed_s=elapsed,
        )
        self._save_cached(result)
        return result
