"""Classification of Manim render failures.

Every failure is bucketed into a stable ``ErrorKind``. Two things depend on this:

* Phase 2 (the render gate) reports pass/fail *by cause*, so we can tell a
  broken corpus from a broken environment. Ten thousand ``LATEX_MISSING``
  failures mean TeX isn't installed, not that the data is bad.
* Phase 7 (the repair loop) routes on it. An ``API_MISUSE`` failure wants the
  relevant docstring injected; a ``TIMEOUT`` wants a simpler scene, not a
  retry of the same code.
"""

from __future__ import annotations

import re
from enum import Enum


class ErrorKind(str, Enum):
    NONE = "none"
    LATEX_MISSING = "latex_missing"        # environment: no `latex` binary
    LATEX_COMPILE = "latex_compile"        # the TeX source itself is invalid
    SYNTAX = "syntax"                      # code never parsed
    IMPORT = "import"                      # wrong/missing module (often ManimGL code)
    NAME = "name"                          # undefined symbol
    API_MISUSE = "api_misuse"              # wrong attribute/argument on a real object
    NO_SCENE = "no_scene"                  # no Scene subclass found
    EMPTY_RENDER = "empty_render"          # ran clean but produced no video
    TIMEOUT = "timeout"                    # exceeded wall clock
    MEMORY = "memory"
    FFMPEG = "ffmpeg"
    LAUNCH = "launch"                       # interpreter/CLI could not be started
    UNKNOWN = "unknown"


#: Ordered most-specific first — the first match wins, so a LaTeX failure is
#: never mistaken for the generic FileNotFoundError it is built on.
_PATTERNS: list[tuple[ErrorKind, re.Pattern[str]]] = [
    # The interpreter has no Manim at all -- a harness pointed at the wrong
    # python. Classed as UNKNOWN it was cached as a verdict on the code, and
    # the next correctly-configured run read "reference fails" from the
    # cache. `manimlib` and `manim.x` are the code's fault and stay IMPORT.
    (ErrorKind.LAUNCH, re.compile(r"No module named '?manim'?(?![\w.])")),
    (ErrorKind.LATEX_MISSING, re.compile(
        r"No such file or directory: '(latex|xelatex|pdflatex|dvisvgm)'|"
        r"(latex|dvisvgm).{0,40}not found", re.I)),
    (ErrorKind.LATEX_COMPILE, re.compile(
        r"LaTeX error|latex_error|Undefined control sequence|"
        r"LaTeX Error:|\.tex.{0,60}failed", re.I)),
    (ErrorKind.SYNTAX,  re.compile(r"SyntaxError|IndentationError|TabError")),
    (ErrorKind.IMPORT,  re.compile(r"ModuleNotFoundError|ImportError|cannot import name")),
    (ErrorKind.NAME,    re.compile(r"NameError")),
    (ErrorKind.API_MISUSE, re.compile(
        r"AttributeError|TypeError|ValueError|KeyError|IndexError|"
        r"got an unexpected keyword argument|takes \d+ positional argument")),
    (ErrorKind.MEMORY,  re.compile(r"MemoryError|Cannot allocate memory|out of memory", re.I)),
    (ErrorKind.FFMPEG,  re.compile(r"ffmpeg|libx264", re.I)),
]

#: Failures caused by the machine, not the code. These must never be counted
#: against a corpus row, and must never be sent to a model for "repair".
ENVIRONMENT_KINDS = frozenset({
    ErrorKind.LATEX_MISSING,
    ErrorKind.MEMORY,
    ErrorKind.FFMPEG,
    ErrorKind.LAUNCH,
})

#: Failures a model can plausibly fix from the traceback alone.
REPAIRABLE_KINDS = frozenset({
    ErrorKind.SYNTAX,
    ErrorKind.IMPORT,
    ErrorKind.NAME,
    ErrorKind.API_MISUSE,
    ErrorKind.LATEX_COMPILE,
    ErrorKind.NO_SCENE,
    ErrorKind.EMPTY_RENDER,
})


def classify(stderr: str, *, timed_out: bool = False) -> ErrorKind:
    """Bucket a failure from its stderr. Returns NONE for empty input."""
    if timed_out:
        return ErrorKind.TIMEOUT
    if not stderr or not stderr.strip():
        return ErrorKind.NONE
    for kind, pattern in _PATTERNS:
        if pattern.search(stderr):
            return kind
    return ErrorKind.UNKNOWN


def is_environment_failure(kind: ErrorKind) -> bool:
    """True if this failure indicts the machine rather than the code."""
    return kind in ENVIRONMENT_KINDS


def is_repairable(kind: ErrorKind) -> bool:
    """True if a model has enough signal in the traceback to attempt a fix."""
    return kind in REPAIRABLE_KINDS


def tail(stderr: str, lines: int = 12) -> str:
    """Last N non-blank stderr lines — what gets fed to the repair model.

    Manim tracebacks are box-drawn and extremely long; the operative error is
    almost always at the bottom. Published work on this task feeds back roughly
    this much and no more.
    """
    kept = [ln for ln in (stderr or "").splitlines() if ln.strip()]
    return "\n".join(kept[-lines:])
