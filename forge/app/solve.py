"""Compute, don't guess: the numbers a scene shows are worked out by Python.

Graded blind (2026-10-10), the commonest way a v1.5 scene is wrong is a
number the model worked out in its head: a mean of 47/8 for a list summing
to 41, a projectile's range of 14.2 m for 10.2, a matrix product off by a
column. A 9B model's arithmetic is the weak link; Python's is not.

So before the scene is written, the model writes a short program that
computes every number the explanation needs from the request's own numbers.
It is run (in a separate process, with a time limit), and its variables are

  * handed to the scene writer as WORKED NUMBERS, and
  * defined at the top of the scene, so the scene can show them with
    f-strings -- ``stage.equation(f"\\bar x = {fmt(mean)}")`` -- instead of
    retyping them.

A number on screen that is neither one of these, nor in the request, nor a
small count, is reported to the critic (``unexplained``), which sends the
draft back for its rewrite.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field

SOLVE_SYSTEM = (
    "A short animation is about to be made to answer a student's question. "
    "Your job is only the numbers. Write a short Python program and nothing "
    "else: every number the explanation will show is a variable, computed "
    "from the question's own numbers (never type a result you worked out "
    "yourself), each with a comment saying what it is. Use plain names "
    "(total, mean, speed_ms) and keep it under 25 lines. You may use math, "
    "Fraction (from fractions) and sympy. End with a variable `answer`. If the "
    "question asks why or what something is, choose a small worked example "
    "and compute its numbers.")

_RUNNER = r"""
import json, math, sys
from fractions import Fraction
try:
    import sympy
except Exception:
    sympy = None
src = sys.stdin.read()
env = {"math": math, "Fraction": Fraction, "sympy": sympy, "__builtins__": __builtins__}
before = set(env)
exec(compile(src, "<solve>", "exec"), env)
out = {}
def plain(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v if math.isfinite(v) else None
    if isinstance(v, Fraction):
        return {"fraction": [v.numerator, v.denominator], "value": float(v)}
    if isinstance(v, (list, tuple)) and 0 < len(v) <= 12:
        xs = [plain(x) for x in v]
        return xs if all(x is not None for x in xs) else None
    if sympy is not None and isinstance(v, sympy.Basic):
        try:
            f = float(v)
            return {"sympy": str(v), "value": f}
        except Exception:
            return {"sympy": str(v)}
    return None
for k, v in env.items():
    if k in before or k.startswith("_"):
        continue
    p = plain(v)
    if p is not None:
        out[k] = p
print(json.dumps(out))
"""


@dataclass
class Solution:
    code: str = ""
    values: dict = field(default_factory=dict)
    comments: dict = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.values) and not self.error


def extract_code(reply: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", reply, re.S)
    code = m.group(1) if m else reply
    # Imports are provided; the runner refuses nothing else, but models
    # import what is already there.
    return "\n".join(line for line in code.splitlines()
                     if not re.match(r"\s*(from\s+\S+\s+)?import\s", line)).strip()


def run(code: str, timeout: float = 10.0) -> Solution:
    sol = Solution(code=code)
    try:
        p = subprocess.run([sys.executable, "-I", "-c", _RUNNER], input=code,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        sol.error = "the program took too long"
        return sol
    if p.returncode != 0:
        last = (p.stderr or "").strip().splitlines()[-1:] or ["error"]
        sol.error = last[0][:200]
        return sol
    try:
        sol.values = json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        sol.error = "no output"
        return sol
    for line in code.splitlines():
        m = re.match(r"\s*([A-Za-z_]\w*)\s*=.*?#\s*(.+)$", line)
        if m:
            sol.comments[m.group(1)] = m.group(2).strip()
    return sol


#: Names the program must not take: the scene's own, and the kit's.
def _reserved() -> set[str]:
    from forge.kit import kit
    return {n for n in dir(kit) if not n.startswith("_")} | {
        "stage", "self", "np", "math", "ax", "p", "plane", "axes", "f", "g"}


def _rename(code: str, names: set[str]) -> str:
    for n in sorted(names, key=len, reverse=True):
        code = re.sub(rf"(?<![\w.]){re.escape(n)}(?!\w)", n + "_val", code)
    return code


def solve(request: str, model, tok, ask=None, max_tokens: int = 700) -> Solution:
    if ask is None:
        from forge.app.pipeline import ask
    reply = ask(model, tok, SOLVE_SYSTEM, f"QUESTION: {request}", max_tokens=max_tokens)
    sol = run(extract_code(reply))
    if sol.ok and (clash := set(sol.values) & _reserved()):
        sol = run(_rename(sol.code, clash))
    if not sol.ok:                       # one retry with the error shown
        reply = ask(model, tok, SOLVE_SYSTEM,
                    f"QUESTION: {request}\n\nYOUR PROGRAM\n{sol.code}\n\nIT FAILED: "
                    f"{sol.error or 'it defined no numbers'}\nWrite it again.",
                    max_tokens=max_tokens)
        sol = run(extract_code(reply))
        if sol.ok and (clash := set(sol.values) & _reserved()):
            sol = run(_rename(sol.code, clash))
    return sol


def _value(v):
    if isinstance(v, dict):
        return v.get("value")
    return v


def _show(v) -> str:
    if isinstance(v, dict) and "fraction" in v:
        p, q = v["fraction"]
        return f"{p}/{q} (= {v['value']:.6g})"
    if isinstance(v, dict) and "sympy" in v:
        return v["sympy"] + (f" (= {v['value']:.6g})" if "value" in v else "")
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, list):
        return "[" + ", ".join(_show(x) for x in v) + "]"
    return str(v)


def facts_text(sol: Solution) -> str:
    """The worked numbers as the scene writer sees them."""
    lines = []
    for k, v in sol.values.items():
        c = sol.comments.get(k, "")
        lines.append(f"  {k} = {_show(v)}" + (f"   # {c}" if c else ""))
    return ("WORKED NUMBERS (computed by running Python; these variables already "
            "exist in the scene -- show them with f-strings and fmt(), e.g. "
            "stage.equation(f\"= {fmt(answer)}\"), and never work out a number "
            "yourself):\n" + "\n".join(lines))


def preamble(sol: Solution) -> str:
    """The program itself, to define the variables at the top of the scene
    (math, Fraction and sympy come with the kit)."""
    return "# worked numbers (computed, forge/app/solve.py)\n" + sol.code.strip()


# -- numbers on screen that nothing explains -----------------------------------

_NUM = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)*(?:\.\d+)?")


def _numbers(s: str) -> list[float]:
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r" \1 / \2 ", s)
    s = re.sub(r"\\[A-Za-z]+", " ", s)           # \times, \sqrt, ...
    s = re.sub(r"\{[^{}]*\}", lambda m: " " + m.group(0)[1:-1] + " ", s)
    out = []
    for m in _NUM.finditer(s):
        t = m.group(0).replace(",", "")
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def _flat(values) -> list[float]:
    out = []
    for v in values:
        if isinstance(v, list):
            out += _flat(v)
            continue
        x = _value(v)
        if isinstance(x, (int, float)):
            out.append(float(x))
        if isinstance(v, dict) and "fraction" in v:
            out += [float(v["fraction"][0]), float(v["fraction"][1])]
    return out


def unexplained(request: str, bodies: list[str], sol: Solution) -> list[tuple[int, str]]:
    """(beat, number) for each number shown on screen that is not a worked
    number (to the precision shown, or as a percentage), not in the request,
    and not a small whole number."""
    from forge.app.critique import shown_texts
    known = _flat(sol.values.values()) + _numbers(request)
    known += [k * 100 for k in known] + [k / 100 for k in known]

    def explained(x: float, shown: str) -> bool:
        if x == int(x) and abs(x) <= 12:
            return True
        dp = len(shown.split(".")[1]) if "." in shown else 0
        tol = 0.5 * 10 ** -dp + 1e-9
        return any(abs(abs(x) - abs(k)) <= tol for k in known)

    out = []
    for n, body in enumerate(bodies, 1):
        for text in shown_texts(body):
            # f-strings are not literals and never reach here: computed.
            stripped = re.sub(r"\{[^{}]*\(.*?\)\}|\{[A-Za-z_]\w*\}", " ", text)
            for tok in _NUM.finditer(re.sub(r"\\[A-Za-z]+", " ", stripped)):
                shown = tok.group(0).replace(",", "")
                try:
                    x = float(shown)
                except ValueError:
                    continue
                if not explained(x, shown):
                    out.append((n, shown))
    seen, uniq = set(), []
    for n, s in out:
        if (n, s) not in seen:
            seen.add((n, s))
            uniq.append((n, s))
    return uniq
