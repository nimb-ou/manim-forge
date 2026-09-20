"""The canonical corpus row.

Six public sources use six different column namings — ``Reviewed Description``
/ ``Code``, ``query`` / ``answer``, ``input`` / ``output``, and raw ``.py``
files with no description at all. Everything normalises to this one shape so
that nothing downstream has to know where a row came from.

Three fields exist because of things measured rather than assumed:

``flavor``
    ManimCE and ManimGL share a name and almost nothing else. Mixing them
    teaches a model to hallucinate across both APIs, so the dialect is recorded
    at ingest and never inferred later.

``n_play_calls``
    ManimBench turned out to be 64% static — code that calls ``self.add`` and
    never animates. Counting animation calls per row is what let us see that,
    and it stays a first-class field so the training mix can be weighted toward
    rows that actually move.

``license``
    Per-row, because the corpus mixes CC BY-NC-SA with Apache-2.0 and the
    project's own share-alike obligation has to be traceable to its sources.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field, asdict


CE = "ce"
GL = "gl"
UNKNOWN = "unknown"

_GL_MARKERS = re.compile(r"\bmanimlib\b|\bfrom\s+manimlib|ShowCreation|TexMobject|TextMobject")
_CE_MARKERS = re.compile(r"\bfrom\s+manim\s+import|\bimport\s+manim\b")


def detect_flavor(code: str) -> str:
    """Which Manim dialect is this?

    ManimGL is checked first: its distinctive names (``ShowCreation``,
    ``TexMobject``) are unambiguous, whereas ``import manim`` appears in both
    worlds often enough to mislead.
    """
    if _GL_MARKERS.search(code):
        return GL
    if _CE_MARKERS.search(code):
        return CE
    return UNKNOWN


def count_play_calls(code: str) -> int:
    return len(re.findall(r"\.play\s*\(", code))


def count_waits(code: str) -> int:
    return len(re.findall(r"\.wait\s*\(", code))


def scene_classes(code: str) -> list[str]:
    """Scene subclasses by AST. Returns [] on unparseable code, which is itself
    a useful signal — such a row can never render."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for b in node.bases:
                nm = b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                if "Scene" in nm:
                    out.append(node.name)
                    break
    return out


def normalized_hash(code: str) -> str:
    """Identity for dedupe, insensitive to formatting and identifier choice.

    The same scene reappears across four datasets with renamed variables and
    reflowed whitespace. Hashing the AST's *structure* — node types only —
    collapses those into one row, which raw-text hashing never would.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Unparseable: fall back to aggressive whitespace normalisation so at
        # least exact-ish duplicates still collapse.
        return "s:" + hashlib.sha256(re.sub(r"\s+", " ", code).strip().encode()).hexdigest()[:20]
    shape = " ".join(type(n).__name__ for n in ast.walk(tree))
    return "a:" + hashlib.sha256(shape.encode()).hexdigest()[:20]


@dataclass
class CorpusRow:
    id: str
    source: str
    license: str
    prompt: str
    code: str

    flavor: str = UNKNOWN
    scene_class: str | None = None
    n_play_calls: int = 0
    n_waits: int = 0
    code_chars: int = 0
    parses: bool = False
    dedupe_key: str = ""

    tags: list[str] = field(default_factory=list)
    render: dict | None = None          # filled by the Phase 2 gate

    @property
    def is_animated(self) -> bool:
        """Does anything actually move? The 3b1b style needs motion; a static
        ``self.add`` diagram is a different (much easier) product."""
        return self.n_play_calls > 0

    @classmethod
    def build(cls, *, source: str, license: str, prompt: str, code: str,
              index: int | None = None,
              tags: list[str] | None = None) -> "CorpusRow":
        """Build a row. `index` is the row's position in its source file.

        Pass `index=None` for generated rows, which have no position in
        anything: the id is then the content hash, which is stable, unique,
        and does not depend on a counter that a caller has to remember to
        advance. `generate_forever.py` passed a literal 0 for every row it
        ever wrote, so 399 distinct scenes in stream.jsonl carried 10 ids
        between them and any join keyed on id silently discarded 97% of
        them.
        """
        from .clean import repair
        code, repair_used = repair(code or "")
        tags = list(tags or [])
        if repair_used:
            tags.append(f"repaired:{repair_used}")
        scenes = scene_classes(code)
        key = normalized_hash(code or "")
        ident = f"{source}:{index:06d}" if index is not None else f"{source}:{key[:12]}"
        return cls(
            id=ident,
            source=source,
            license=license,
            prompt=(prompt or "").strip(),
            code=(code or "").strip(),
            flavor=detect_flavor(code),
            scene_class=scenes[-1] if scenes else None,
            n_play_calls=count_play_calls(code),
            n_waits=count_waits(code),
            code_chars=len(code or ""),
            parses=bool(scenes) or _parses(code),
            dedupe_key=key,
            tags=tags,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _parses(code: str) -> bool:
    try:
        ast.parse(code or "")
        return True
    except SyntaxError:
        return False
