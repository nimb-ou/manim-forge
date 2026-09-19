"""Deterministic repairs — fixes that need no model call at all.

Three of twenty baseline failures were ``empty_render``: code that is entirely
correct and simply produces no video, because it calls ``self.add`` and never
``self.play`` or ``self.wait``. Manim renders zero frames and writes nothing.

That failure traces straight back to the corpus. ManimBench is 64% static, so a
model trained on it writes static scenes. Fixing it needs no intelligence —
just a trailing ``self.wait()`` — so it should never cost a generation.

Every rule here is conservative: it only fires on a specific detectable
condition, and the result is checked to still parse before being accepted.
"""

from __future__ import annotations

import ast
import re


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def needs_hold(code: str) -> bool:
    """Scene adds mobjects but never animates or waits -> zero-length video."""
    has_motion = re.search(r"self\.(play|wait)\s*\(", code) is not None
    has_content = re.search(r"self\.add\s*\(", code) is not None
    return has_content and not has_motion


def add_hold(code: str, seconds: float = 1.0) -> str:
    """Append ``self.wait(...)`` to the end of ``construct``.

    Indentation is copied from the last statement inside construct rather than
    assumed, since generated code mixes 4- and 8-space bodies.
    """
    lines = code.rstrip().split("\n")
    indent = "        "
    for ln in reversed(lines):
        if ln.strip() and not ln.strip().startswith("#"):
            m = re.match(r"^(\s+)\S", ln)
            if m:
                indent = m.group(1)
            break
    return "\n".join(lines + [f"{indent}self.wait({seconds})"])


def strip_prose(code: str) -> str:
    """Remove a leading English sentence the model wrote outside a fence."""
    lines = code.split("\n")
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(from|import|class|def|#|@)", ln):
            return "\n".join(lines[i:])
    return code


#: (name, applies_when, transform)
RULES = [
    ("strip_prose", lambda c: not _parses(c), strip_prose),
    ("add_hold", needs_hold, add_hold),
]


def lint(code: str) -> tuple[str, list[str]]:
    """Apply every rule whose condition holds. Returns (code, rules_applied).

    A rule whose output fails to parse is discarded, so linting can only ever
    leave the code the same or better.
    """
    applied = []
    for name, condition, transform in RULES:
        try:
            if not condition(code):
                continue
            candidate = transform(code)
        except Exception:
            continue
        if candidate != code and _parses(candidate):
            code, _ = candidate, applied.append(name)
    return code, applied
