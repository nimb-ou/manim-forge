"""Arithmetic on screen, checked: does every "a = b" a scene shows agree?

A 7B writes "15% of £80 = £10" as fluently as "= £12". Most of the numbers
a scene shows sit in stage.equation / caption / label strings as chains of
equals signs. Where two neighbouring sides of a chain are pure arithmetic
(numbers, + - × ÷ · /, powers, fractions, percentages, a currency sign) both
are evaluated, and a mismatch is reported. Anything with a letter in it is
skipped -- "a^2 + b^2 = c^2" is algebra, not a slip -- so this only ever
catches real arithmetic errors, never style.
"""
from __future__ import annotations

import ast
import math
import re
import warnings

_CALLS = re.compile(r"stage\.(?:equation|caption|label|title)\s*\(")
_STR = re.compile(r"""(?:[rRbBuUfF]{0,2})("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')""")


def _clean(side: str) -> str | None:
    """An arithmetic expression in Python syntax, or None if it is not pure
    arithmetic."""
    s = side.strip()
    # A lone percentage ("15% = 4 + 2") is shorthand for "15% of the bill";
    # a lone run of 0s and 1s ("1101 = -8 + 4 + 0 + 1") is binary.
    if re.fullmatch(r"\d+(\.\d+)?\s*\\?%", s) or re.fullmatch(r"[01]{3,}", s):
        return None
    s = re.sub(r"\\(?:d|t)?frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"((\1)**0.5)", s)
    for a, b in ((r"\times", "*"), (r"\cdot", "*"), (r"\div", "/"), ("×", "*"),
                 ("÷", "/"), ("·", "*"), ("−", "-"), (r"\left", ""), (r"\right", ""),
                 (r"\,", ""), (r"\!", ""), (r"\%", "%"), ("^", "**"), ("{", "("),
                 ("}", ")"), ("£", ""), ("$", ""), ("€", ""), (r"\$", ""), ("\\", "")):
        s = s.replace(a, b)
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)            # 3,400 -> 3400
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", s)
    s = s.strip().rstrip(".")
    if not s or re.search(r"[A-Za-z_]", s) or not re.search(r"\d", s):
        return None
    if not re.fullmatch(r"[\d\s.+\-*/()]+", s):
        return None
    return s


def _value(expr: str) -> float | None:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    ok = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub,
          ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)
    if not all(isinstance(n, ok) for n in ast.walk(tree)):
        return None
    try:
        v = eval(compile(tree, "<arith>", "eval"), {"__builtins__": {}})  # noqa: S307
    except (ZeroDivisionError, OverflowError, ValueError, TypeError):
        return None
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


def _agree(a: float, b: float) -> bool:
    # Shown values are rounded: 1/3 = 0.33, 22/7 = 3.14.
    return abs(a - b) <= max(0.011, 0.006 * max(abs(a), abs(b)))


def chain_errors(text: str) -> list[str]:
    """Mismatched neighbours in one "a = b = c" string."""
    if "=" not in text or any(op in text for op in ("<", ">", "≈", "\\approx", "\\ne")):
        return []
    sides = [s for s in re.split(r"=", text)]
    vals = [(_value(c) if (c := _clean(s)) else None) for s in sides]
    out = []
    for (s1, v1), (s2, v2) in zip(zip(sides, vals), zip(sides[1:], vals[1:])):
        if v1 is not None and v2 is not None and not _agree(v1, v2):
            out.append(f"{s1.strip()} = {s2.strip()}")
    return out


def arithmetic_errors(code: str) -> list[str]:
    """Every arithmetic mismatch in the strings a scene puts on screen."""
    out = []
    for m in _CALLS.finditer(code):
        depth, i = 1, m.end()
        while i < len(code) and depth:
            depth += {"(": 1, ")": -1}.get(code[i], 0)
            i += 1
        for sm in _STR.finditer(code[m.end(): i - 1]):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    s = ast.literal_eval(sm.group(0))
            except (SyntaxError, ValueError):
                continue
            if isinstance(s, str):
                out += chain_errors(s)
    return out
