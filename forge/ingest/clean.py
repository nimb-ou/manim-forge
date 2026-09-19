"""Repair of corrupted source text.

The `thanhkt` set — the largest single public source at 4.4k rows — ships code
whose newlines were escaped twice somewhere in its pipeline. Rows begin with a
literal backslash-n rather than a line break, so Python reports "unexpected
character after line continuation character" and 60% of the raw corpus fails to
parse.

This is worth repairing rather than discarding: a row that only needs its
escaping undone is a perfectly good training example underneath. Repairs are
attempted conservatively and only kept when they demonstrably help — a repair
that does not make the code parse is thrown away, so this can never make a row
worse than it started.
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


def _strip_literal_newline_prefix(code: str) -> str:
    """Drop a leading literal ``\\n`` that isn't a real line break."""
    c = code.lstrip()
    while c.startswith("\\n"):
        c = c[2:].lstrip()
    return c


def _unescape_newlines(code: str) -> str:
    """Turn literal ``\\n`` / ``\\t`` sequences into real whitespace.

    Deliberately narrow. A blanket ``unicode_escape`` decode would also mangle
    the LaTeX in MathTex strings, where a single backslash is meaningful and
    must survive untouched.
    """
    return code.replace("\\n", "\n").replace("\\t", "    ")


def _dedent_body(code: str) -> str:
    """Normalise over-indented method bodies.

    Some rows indent statements inside ``construct`` by eight or twelve spaces
    inconsistently, which is a syntax error rather than a style problem.
    """
    lines = code.split("\n")
    out, fixed = [], False
    for ln in lines:
        m = re.match(r"^(\s{12,})(\S)", ln)
        if m and not fixed:
            out.append("        " + ln.lstrip())
        else:
            out.append(ln)
    return "\n".join(out)


#: Tried in order, cheapest and safest first.
_STRATEGIES = [
    ("strip_prefix", _strip_literal_newline_prefix),
    ("unescape", lambda c: _unescape_newlines(_strip_literal_newline_prefix(c))),
    ("dedent", lambda c: _dedent_body(_unescape_newlines(_strip_literal_newline_prefix(c)))),
]


def repair(code: str) -> tuple[str, str | None]:
    """Return ``(code, strategy_used)``.

    Already-valid code is returned untouched. If no strategy makes it parse,
    the original is returned with ``None`` — the render gate will deal with it,
    and we have not silently corrupted anything.
    """
    if not code:
        return code, None
    if _parses(code):
        return code, None

    for name, fn in _STRATEGIES:
        try:
            candidate = fn(code)
        except Exception:
            continue
        if _parses(candidate):
            return candidate, name

    return code, None
