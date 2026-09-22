"""Check a scene's calls against the installed Manim, before rendering it.

Run 17's failures are not typos. Where the control wrote ``Dodecahedron()``
it hand-built a ``Polyhedron`` from vertex coordinates; where the control
drew a ``SurroundingRectangle`` it tried ``table[0, :].set_color(...)``. All
23 of its remaining benchmark failures are `api_misuse`, with the same error
at every one of four repair rounds.

Every one of those costs a render to discover -- up to 120 seconds of
subprocess, LaTeX and ffmpeg to learn that a keyword argument does not
exist. Manim is installed locally, so the same fact is available from
`inspect.signature` in microseconds.

This is deliberately narrow. It reports only what it can be *certain* of:

- a name that Manim does not export at all
- a keyword argument a constructor does not accept, where the callable takes
  no ``**kwargs``

It does not guess types, does not check positional arity (Manim's own
classes are liberal about it), and says nothing when it cannot resolve a
name. A checker that cries wolf is one whose output gets ignored, and this
one is meant to be trusted enough to put in a repair prompt.

Suggestions come from difflib against the real export list, which is how
`Polyhedron` should surface `Dodecahedron`.
"""
from __future__ import annotations

import ast
import builtins
import difflib
import inspect
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Finding:
    line: int
    name: str
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.message}"


@lru_cache(maxsize=1)
def _manim():
    import manim
    return manim


@lru_cache(maxsize=1)
def _exports() -> frozenset[str]:
    m = _manim()
    return frozenset(n for n in dir(m) if not n.startswith("_"))


@lru_cache(maxsize=512)
def _accepted(name: str) -> tuple[frozenset[str], bool] | None:
    """(keyword names, takes **kwargs) for a Manim callable, or None.

    Walks the MRO, because Manim's mobjects take most of their keywords from
    base classes -- checking only the leaf __init__ would reject almost
    every correct scene.
    """
    obj = getattr(_manim(), name, None)
    if obj is None:
        return None
    target = obj.__init__ if inspect.isclass(obj) else obj
    if not callable(target):
        return None
    names: set[str] = set()
    var_kw = False
    bases = inspect.getmro(obj) if inspect.isclass(obj) else [obj]
    for base in bases:
        fn = base.__init__ if inspect.isclass(base) else base
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return None
        for p in sig.parameters.values():
            if p.kind is inspect.Parameter.VAR_KEYWORD:
                var_kw = True
            elif p.kind in (inspect.Parameter.KEYWORD_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD):
                names.add(p.name)
    return frozenset(names), var_kw


def check(code: str, max_findings: int = 6) -> list[Finding]:
    """Certain problems only. Empty list means 'nothing I can prove wrong'."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    try:
        exports = _exports()
    except Exception:                                         # noqa: BLE001
        return []                       # no Manim here; say nothing

    # Anything bound in the file shadows the Manim name of the same spelling.
    local: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            local.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            local.add(node.name)
        elif isinstance(node, ast.alias):
            local.add((node.asname or node.name).split(".")[0])

    out: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        # `dir(__builtins__)` returns dict methods here, not builtin names --
        # inside a module __builtins__ is the dict, not the module. That let
        # `range` and `enumerate` through as "Manim has no 'range'. Did you
        # mean Triangle?", which is the cry-wolf failure this file's
        # docstring warns about, produced by this file.
        if name in local or hasattr(builtins, name):
            continue
        if name not in exports:
            near = difflib.get_close_matches(name, exports, n=3, cutoff=0.7)
            hint = f" Did you mean {', '.join(near)}?" if near else ""
            out.append(Finding(node.lineno, name,
                               f"Manim has no {name!r}.{hint}"))
            continue
        acc = _accepted(name)
        if acc is None:
            continue
        accepted, var_kw = acc
        if var_kw:
            continue
        for kw in node.keywords:
            if kw.arg and kw.arg not in accepted:
                near = difflib.get_close_matches(kw.arg, accepted, n=2,
                                                 cutoff=0.7)
                hint = f" Did you mean {', '.join(near)}?" if near else ""
                out.append(Finding(
                    node.lineno, name,
                    f"{name}() takes no {kw.arg!r} argument.{hint}"))
        if len(out) >= max_findings:
            break
    return out[:max_findings]


def briefing(code: str) -> str:
    """The findings as a block for a repair prompt, or '' if there are none."""
    found = check(code)
    if not found:
        return ""
    return ("Checked against the installed Manim before rendering:\n"
            + "\n".join(f"  {f}" for f in found))
