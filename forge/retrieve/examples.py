"""Example retrieval — few-shot context drawn from verified scenes.

Retrieval-augmented generation for code works far better over *working
examples* than over documentation. A model handed prose about how Table works
still has to invent the call; a model handed a Table scene that demonstrably
rendered has a pattern to adapt.

That distinction matters more here than usual, because the published work on
this task found dense API documentation actively degraded smaller models by
saturating their context. Examples are denser in useful signal per token.

Everything retrieved has passed the render gate, so a retrieved example is
never a pattern that fails to run. Embeddings come from nomic-embed-text via
Ollama: local, free, and already installed.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

EMBED_MODEL = "nomic-embed-text"


def embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed via the local Ollama server. Batched in one call per text because
    Ollama's embedding endpoint takes a single prompt."""
    import urllib.request

    out = []
    for t in texts:
        body = json.dumps({"model": model, "prompt": t}).encode()
        req = urllib.request.Request("http://localhost:11434/api/embeddings",
                                     data=body,
                                     headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=120))
        out.append(d["embedding"])
    return out


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Example:
    prompt: str
    code: str
    source: str
    n_play_calls: int


class ExampleIndex:
    """A small in-memory index. Brute force on purpose.

    A few thousand vectors compare in milliseconds, so a vector database would
    add a dependency and an operational surface to solve a problem we do not
    have. Revisit at a hundred thousand rows, not before.
    """

    def __init__(self) -> None:
        self.examples: list[Example] = []
        self.vectors: list[list[float]] = []

    @classmethod
    def build(cls, train_jsonl: Path, limit: int | None = None,
              min_play_calls: int = 1) -> "ExampleIndex":
        idx = cls()
        seen = set()
        rows = []
        for line in train_jsonl.open():
            r = json.loads(line)
            meta = r.get("meta", {})
            # The training file repeats animated rows for weighting; an index
            # wants each scene once.
            if meta.get("id") in seen:
                continue
            seen.add(meta.get("id"))
            if meta.get("n_play_calls", 0) < min_play_calls:
                continue
            msgs = {m["role"]: m["content"] for m in r["messages"]}
            rows.append(Example(
                prompt=msgs.get("user", ""),
                code=msgs.get("assistant", ""),
                source=meta.get("source", "?"),
                n_play_calls=meta.get("n_play_calls", 0),
            ))
        if limit:
            rows = rows[:limit]

        idx.examples = rows
        idx.vectors = [_norm(v) for v in embed([e.prompt for e in rows])]
        return idx

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for e, v in zip(self.examples, self.vectors):
                f.write(json.dumps({"prompt": e.prompt, "code": e.code,
                                    "source": e.source,
                                    "n_play_calls": e.n_play_calls,
                                    "vec": v}) + "\n")

    @classmethod
    def load(cls, path: Path) -> "ExampleIndex":
        idx = cls()
        for line in path.open():
            d = json.loads(line)
            idx.examples.append(Example(d["prompt"], d["code"], d["source"],
                                        d["n_play_calls"]))
            idx.vectors.append(d["vec"])
        return idx

    def search(self, query: str, k: int = 3) -> list[tuple[float, Example]]:
        qv = _norm(embed([query])[0])
        scored = sorted(
            ((_dot(qv, v), e) for v, e in zip(self.vectors, self.examples)),
            key=lambda t: -t[0],
        )
        return scored[:k]

    def as_fewshot(self, query: str, k: int = 2, max_chars: int = 2600) -> str:
        """Retrieved examples formatted for a prompt, within a char budget.

        Budgeted deliberately: context saturation is a measured failure mode
        for small models, so two good examples beat five mediocre ones.
        """
        parts = []
        used = 0
        for score, ex in self.search(query, k=k * 2):
            block = f"# Request: {ex.prompt}\n{ex.code}\n"
            if used + len(block) > max_chars:
                continue
            parts.append(block)
            used += len(block)
            if len(parts) >= k:
                break
        if not parts:
            return ""
        return ("Here are verified examples of similar animations:\n\n"
                + "\n".join(parts))
