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


#: Standard-library modules generated scenes reach for and forget to import.
#: These surface as NameError and look like API misuse in the failure
#: breakdown, but no model call is needed — the import is mechanical.
_STDLIB = ("random", "math", "itertools", "collections", "fractions", "cmath")


def missing_stdlib_imports(code: str) -> list[str]:
    """Modules used by attribute access but never imported.

    Deliberately narrow: only a module-style `name.attr(` usage counts, so a
    local variable called `random` is not mistaken for the module. `np` is
    excluded because `from manim import *` already provides numpy as np.
    """
    needed = []
    for mod in _STDLIB:
        # Attribute access, call or not: `math.pi` is as common as `math.sin(`
        # in this corpus and an earlier version requiring a call missed it.
        used = re.search(rf"(?<![\w.]){mod}\.[A-Za-z_]\w*", code)
        imported = re.search(rf"^\s*(import\s+{mod}\b|from\s+{mod}\s+import)",
                             code, re.M)
        # A local of the same name shadows the module, so importing would be
        # wrong as well as useless.
        shadowed = re.search(rf"^\s*{mod}\s*=", code, re.M)
        if used and not imported and not shadowed:
            needed.append(mod)
    return needed


def add_stdlib_imports(code: str) -> str:
    """Insert missing imports after the manim import, preserving order."""
    mods = missing_stdlib_imports(code)
    if not mods:
        return code
    lines = code.split("\n")
    insert_at = 0
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(from|import)\s", ln):
            insert_at = i + 1
    for mod in reversed(mods):
        lines.insert(insert_at, f"import {mod}")
    return "\n".join(lines)


#: A bare ``%`` inside a Tex/MathTex string starts a LaTeX comment, so the
#: rest of the line -- the closing brace included -- vanishes and dvisvgm
#: fails with an unhelpful "installation does not support converting PDF to
#: SVG". Two of my own gold scenes shipped with it and neither failed until
#: render time.
#:
#: Scope, measured rather than assumed. Across all 4,515 scenes on disk --
#: the whole scraped corpus, every synthetic row, and the 44 gold scenes --
#: this rule fires **zero** times. It was written for a bug I introduced and
#: have since fixed; it is not a failure mode the models produce.
#:
#: It also covers less than the obvious case suggests. Only *string literals*
#: are inspected, so the form that actually bit me, ``f"{x:.1%}"``, is
#: invisible to it: an f-string is a JoinedStr, and in any case the ``%``
#: there is produced at runtime by the format spec, not present in the
#: source, so there is nothing to escape. That one has no mechanical repair
#: -- the fix is ``f"{x*100:.1f}\\%"`` -- and it does not occur in the
#: corpus either. Kept because it is cheap and correct for what it covers;
#: recorded here so nobody credits it with work it is not doing.


def _tex_string_args(code: str):
    """Every string literal passed positionally to a Tex/MathTex call."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in ("Tex", "MathTex"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out.append(arg)
    return out


def has_bare_percent(code: str) -> bool:
    """True if a Tex literal contains an unescaped ``%``."""
    return any(re.search(r"(?<!\\)%", a.value) for a in _tex_string_args(code))


def escape_tex_percent(code: str) -> str:
    """Escape bare percents inside Tex/MathTex literals only.

    Edits each literal's own source span, so a ``%`` in a comment, a format
    spec, or an ordinary string is left alone. Multi-line literals are
    skipped rather than guessed at.
    """
    lines = code.splitlines(keepends=True)
    offsets, total = [], 0
    for ln in lines:
        offsets.append(total)
        total += len(ln)

    edits = []
    for node in _tex_string_args(code):
        if node.lineno != node.end_lineno:
            continue
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        seg = code[start:end]
        # In a raw literal one backslash is one backslash. In an ordinary one
        # a single "\\%" is an invalid escape -- Python keeps it, but warns --
        # so the replacement has to be doubled there.
        raw = seg[:2].lower().startswith("r")
        repl = r"\\%" if raw else r"\\\\%"
        fixed = re.sub(r"(?<!\\)%", repl, seg)
        if fixed != seg:
            edits.append((start, end, fixed))
    for start, end, fixed in sorted(edits, reverse=True):
        code = code[:start] + fixed + code[end:]
    return code


#: Manim's own namespace, forgotten. 1,070 of the 1,920 corpus rows that
#: failed the gate declare a Scene subclass with no manim import at all, and
#: **zero** passing rows do. Most of those 1,070 are truncated fragments that
#: do not even parse -- the missing import is a symptom there, not the cause.
#: The 103 that *do* parse and failed with NameError are a different matter:
#: for those the import is the whole defect, and supplying it costs no model
#: call.
#:
#: Deliberately narrow. It fires only when the file parses, names a manim
#: base class, and imports neither manim nor manimlib -- so a ManimGL row
#: (a different, incompatible library) is never handed a Community import,
#: and a fragment that cannot be parsed is left for strip_prose or the model.

_SCENE_BASES = ("Scene", "ThreeDScene", "MovingCameraScene", "ZoomedScene",
                "VectorScene", "LinearTransformationScene", "SpecialThreeDScene")


def _imports_a_manim(code: str) -> bool:
    return bool(re.search(r"^\s*(from\s+manim(lib|_imports|ce)?\b|import\s+manim)",
                          code, re.M))


def needs_manim_import(code: str) -> bool:
    """Declares a manim Scene subclass but never imports the namespace."""
    if _imports_a_manim(code):
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False       # a fragment: not ours to fix
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            name = getattr(base, "id", None) or getattr(base, "attr", None)
            if name in _SCENE_BASES:
                return True
    return False


def add_manim_import(code: str) -> str:
    """Insert ``from manim import *`` above the first statement.

    Above the *first statement*, not at line 0: a module docstring has to stay
    first to remain a docstring, and `from __future__` imports are a syntax
    error anywhere else.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    lines = code.split("\n")
    at = 0
    for node in tree.body:
        is_docstring = (isinstance(node, ast.Expr)
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str))
        is_future = (isinstance(node, ast.ImportFrom)
                     and node.module == "__future__")
        if is_docstring or is_future:
            at = node.end_lineno
            continue
        break
    lines.insert(at, "from manim import *")
    return "\n".join(lines)


RULES = [
    ("strip_prose", lambda c: not _parses(c), strip_prose),
    ("add_manim_import", needs_manim_import, add_manim_import),
    ("add_imports", lambda c: bool(missing_stdlib_imports(c)), add_stdlib_imports),
    ("add_hold", needs_hold, add_hold),
    ("escape_percent", has_bare_percent, escape_tex_percent),
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
