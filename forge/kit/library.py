"""The hand-written scenes as a searchable library.

Every teacher scene (forge/kit/teacher/batch_*.json) was looked at on a
contact sheet and its maths checked. Rather than hoping a 7B absorbs ~930
scenes through its weights and then invents a coherent arc from one
sentence, the one-shot generator (forge/app/oneshot.py) is shown the nearest
ones and adapts them.

Retrieval scores the request against each scene's request and beat
intents twice -- TF-IDF over word unigrams and bigrams, and a small
sentence embedding (bge-small) -- and adds the two, each standardised over
the library, the embedding counted double. TF-IDF alone matched "how do you
add 3/4 and 1/6?" to "which is bigger, 3/4 or 5/8?"; the embedding finds
"how do you add 1/3 and 1/4?", but alone matched a line's slope to rounding
2.47. Without torch (CI), or with FORGE_RETRIEVAL=tfidf, it is TF-IDF only.

Scenes touching a held-out topic are left out of the library entirely, so a
held-out evaluation can never be answered by copying a scene about the same
thing.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENE_DIR = ROOT / "forge" / "kit" / "teacher"     # tracked copy of data/kit/claude_scenes

_STOP = set("""a an the of to and or in on at by for with is are be it its this
that what why how do does did you your i me my we can show shown from as into
when which there their than then so if""".split())


def _words(text: str) -> list[str]:
    # "%" and "×" carry the question type ("15% of 240", "7 × 8").
    text = text.lower().replace("%", " percent ").replace("×", " times ")
    w = [x for x in re.findall(r"[a-z0-9]+(?:[./][0-9]+)?", text)
         if x not in _STOP]
    return w + [f"{a}_{b}" for a, b in zip(w, w[1:])]


@lru_cache(maxsize=1)
def scenes() -> tuple[dict, ...]:
    """Every distinct teacher scene, held-out topics removed."""
    from forge.evaluate.heldout_guard import touches_heldout
    seen, out = set(), []
    for f in sorted(SCENE_DIR.glob("batch_*.json")):
        data = json.loads(f.read_text())
        for s in data if isinstance(data, list) else data.get("scenes", []):
            if not isinstance(s, dict) or not s.get("beats"):
                continue
            key = " ".join(s["request"].lower().split())
            if key in seen:
                continue
            text = s["request"] + " " + " ".join(b["intent"] for b in s["beats"])
            if touches_heldout(text):
                continue
            seen.add(key)
            out.append({"request": s["request"], "beats": s["beats"],
                        "source": f.name})
    return tuple(out)


@lru_cache(maxsize=1)
def _index():
    docs = [Counter(_words(s["request"]) * 2
                    + _words(" ".join(b["intent"] for b in s["beats"])))
            for s in scenes()]
    df = Counter(t for d in docs for t in d)
    n = len(docs)
    idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
    vecs = []
    for d in docs:
        v = {t: (1 + math.log(c)) * idf[t] for t, c in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({t: x / norm for t, x in v.items()})
    return idf, vecs


EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_QUERY = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _embedder():
    """(tokenizer, model, library matrix), or None without torch."""
    if os.environ.get("FORGE_RETRIEVAL") == "tfidf":
        return None
    try:
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
        model = AutoModel.from_pretrained(EMBED_MODEL).eval()
    except Exception:                                         # noqa: BLE001
        return None
    docs = [s["request"] + ". " + "; ".join(b["intent"] for b in s["beats"])
            for s in scenes()]
    key = hashlib.sha256("\n".join(docs).encode()).hexdigest()[:16]
    cache = ROOT / "data" / "kit" / f"library_emb_{key}.npy"
    if cache.exists():
        mat = np.load(cache)
    else:
        mat = _embed(tok, model, docs)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, mat)
    return tok, model, mat


def _embed(tok, model, texts: list[str]):
    import numpy as np
    import torch
    out = []
    for i in range(0, len(texts), 64):
        b = tok(texts[i:i + 64], padding=True, truncation=True,
                max_length=128, return_tensors="pt")
        with torch.no_grad():
            h = model(**b).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h, dim=-1).numpy())
    return np.concatenate(out).astype("float32")


def _z(xs: list[float]) -> list[float]:
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) or 1.0
    return [(x - m) / sd for x in xs]


def similar(text: str, k: int = 2, exclude: set[str] | None = None
            ) -> list[tuple[float, dict]]:
    """The ``k`` scenes nearest ``text``, best first, as (score, scene).

    The score is the TF-IDF cosine, or with the embedder the sum of both
    similarities standardised over the library. ``exclude`` holds requests
    to skip -- a scene's own request when building training rows, so a row
    is never shown its own answer.
    """
    idf, vecs = _index()
    q = Counter(_words(text))
    qv = {t: (1 + math.log(c)) * idf.get(t, 0.0) for t, c in q.items()}
    norm = math.sqrt(sum(x * x for x in qv.values())) or 1.0
    score = [sum(x * v.get(t, 0.0) for t, x in qv.items()) / norm for v in vecs]
    emb = _embedder()
    if emb is not None:
        tok, model, mat = emb
        e = (mat @ _embed(tok, model, [EMBED_QUERY + text])[0]).tolist()
        score = [a + 2 * b for a, b in zip(_z(score), _z(e))]
    ex = {" ".join(e.lower().split()) for e in (exclude or ())}
    scored = [(c, s) for c, s in zip(score, scenes())
              if " ".join(s["request"].lower().split()) not in ex]
    scored.sort(key=lambda p: -p[0])
    return scored[:k]


def as_body(scene: dict) -> str:
    """A scene in the one-shot output format: the construct() body with a
    ``# beat N: intent`` and a ``# say: narration`` line above each beat."""
    out = []
    for n, b in enumerate(scene["beats"], 1):
        out.append(f"# beat {n}: {b['intent']}")
        if b.get("narration"):
            out.append(f"# say: {b['narration']}")
        out.append(b["code"].strip("\n"))
        out.append("")
    return "\n".join(out).rstrip() + "\n"
