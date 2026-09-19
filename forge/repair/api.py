"""Ground truth about the Manim API, by introspection.

The benchmark's dominant failure is ``api_misuse``: the model knows ``Table``
exists and then invents ``include_row_labels=True``. Telling it the real
signature fixes that — but only if the signature we tell it is correct.

So we do not scrape documentation. Manim is installed in this environment, so
``inspect`` can be asked directly. That answer is true for the exact version we
render against, never goes stale, and covers undocumented classes too.

The same trick handles invented names. When the model writes ``PointCloud``,
which does not exist, we can enumerate everything Manim *does* export and offer
the nearest real names instead of a bare "not defined".
"""

from __future__ import annotations

import ast
import difflib
import inspect
import re
from functools import lru_cache


@lru_cache(maxsize=1)
def _manim_namespace() -> dict:
    import manim
    return {n: getattr(manim, n) for n in dir(manim) if not n.startswith("_")}


@lru_cache(maxsize=1)
def _public_names() -> list[str]:
    return sorted(_manim_namespace())


def resolve(name: str):
    return _manim_namespace().get(name)


def exists(name: str) -> bool:
    return name in _manim_namespace()


def suggest(name: str, n: int = 4) -> list[str]:
    """Real Manim names closest to an invented one.

    Cutoff is deliberately loose: a wrong-but-plausible suggestion still points
    the model at the right region of the API, whereas no suggestion leaves it
    to guess again.
    """
    return difflib.get_close_matches(name, _public_names(), n=n, cutoff=0.6)


#: Per-signature budget. Manim's Table has 17 keyword arguments and runs to
#: ~900 characters; six of those would saturate a small model's context, which
#: the published work found actively degrades repair quality.
MAX_SIG_CHARS = 420


def signature_of(name: str) -> str | None:
    """``ClassName(arg, kwarg=default, ...)`` for a real Manim object."""
    obj = resolve(name)
    if obj is None:
        return None
    try:
        target = obj.__init__ if inspect.isclass(obj) else obj
        sig = inspect.signature(target)
        params = [str(p) for p_name, p in sig.parameters.items()
                  if p_name not in ("self", "kwargs", "args")]
        rendered = f"{name}({', '.join(params)})"
        if len(rendered) > MAX_SIG_CHARS:
            # Keep whole parameters rather than cutting mid-token, so what the
            # model reads is always valid syntax it can copy.
            kept, total = [], len(name) + 1
            for prm in params:
                if total + len(prm) + 2 > MAX_SIG_CHARS:
                    break
                kept.append(prm); total += len(prm) + 2
            rendered = f"{name}({', '.join(kept)}, ...)"
        return rendered
    except (ValueError, TypeError):
        return f"{name}(...)"


def summary_of(name: str, max_chars: int = 220) -> str | None:
    obj = resolve(name)
    doc = inspect.getdoc(obj) if obj is not None else None
    if not doc:
        return None
    first = doc.strip().split("\n\n")[0].replace("\n", " ")
    return first[:max_chars]


def names_used(code: str) -> list[str]:
    """Capitalised identifiers the code calls — i.e. candidate Manim classes."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return sorted(set(re.findall(r"\b([A-Z][A-Za-z0-9_]{2,})\s*\(", code)))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            nm = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
            if nm and nm[:1].isupper():
                out.add(nm)
    return sorted(out)


# Names the traceback itself blames, which matter more than the rest.
_BLAME = [
    re.compile(r"name '([A-Za-z_][\w]*)' is not defined"),
    re.compile(r"unexpected keyword argument '([\w]+)'"),
    re.compile(r"has no attribute '([\w]+)'"),
    re.compile(r"^KeyError: '([\w]+)'", re.M),
]


def blamed_names(stderr: str) -> list[str]:
    out = []
    for pat in _BLAME:
        out.extend(pat.findall(stderr or ""))
    return list(dict.fromkeys(out))


def api_briefing(code: str, stderr: str, max_entries: int = 6) -> str:
    """The documentation fragment injected into a repair prompt.

    Deliberately short. The paper this borrows from found that dumping full API
    documentation *hurt* smaller models by saturating their context — so this
    prioritises names the traceback actually blames, then fills remaining slots
    with other Manim classes the code touched.
    """
    blamed = blamed_names(stderr)
    used = names_used(code)

    lines: list[str] = []

    # 1. Invented names -> what actually exists.
    for nm in blamed:
        if nm[:1].isupper() and not exists(nm):
            alts = suggest(nm)
            if alts:
                lines.append(f"`{nm}` does not exist in Manim. Closest real names: "
                             + ", ".join(f"`{a}`" for a in alts))
            else:
                lines.append(f"`{nm}` does not exist in Manim.")

    # 2. Real classes whose signature the model got wrong.
    shown = set()
    for nm in used:
        if len(lines) >= max_entries:
            break
        if not exists(nm) or nm in shown:
            continue
        sig = signature_of(nm)
        if not sig:
            continue
        shown.add(nm)
        entry = f"{sig}"
        doc = summary_of(nm, 140)
        if doc:
            entry += f"\n    # {doc}"
        lines.append(entry)

    if not lines:
        return ""
    return "Relevant Manim API (this is the real signature, use only these arguments):\n" + \
           "\n".join(f"  {l}" for l in lines)
