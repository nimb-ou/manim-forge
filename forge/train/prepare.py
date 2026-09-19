"""Assemble the final training mix and split it honestly.

Three sources feed the model, and they are not interchangeable:

* **scraped + gated** — teaches the Manim API. Large, mediocre, verified.
* **synthetic** — teaches beat structure and breadth of topic. Growing daily.
* **gold** — teaches 3b1b style. Tiny, hand-authored, weighted heavily.

The split is by *prompt*, not by row. Splitting randomly across rows would put
near-identical scenes on both sides — the corpus is full of paraphrases — and
the eval score would then measure memorisation. ManimBench's own test rows are
excluded outright, since the benchmark we report against uses them.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

SYSTEM = (
    "You are an expert Manim Community Edition developer. "
    "Given a description of an animation, write complete, runnable Python code "
    "using `from manim import *` and a single Scene subclass. "
    "Always animate with self.play(...) and end with self.wait(). "
    "Output only code in one ``` block. No explanation."
)


def _key(prompt: str) -> str:
    """Coarse prompt identity for grouping paraphrases together."""
    return re.sub(r"[^a-z0-9 ]", "", prompt.lower())[:90]


def _example(prompt: str, code: str, meta: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": f"```python\n{code.strip()}\n```"},
        ],
        "meta": meta,
    }


def load_gated(train_jsonl: Path) -> list[dict]:
    """Verified rows already exported by the gate, deduplicated by id."""
    out, seen = [], set()
    for line in train_jsonl.open():
        r = json.loads(line)
        rid = r.get("meta", {}).get("id")
        if rid in seen:
            continue
        seen.add(rid)
        out.append(r)
    return out


def load_synthetic(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.open():
        r = json.loads(line)
        if not r.get("ok"):
            continue
        out.append(_example(r["prompt"], r["code"], {
            "id": r["id"], "source": "synthetic", "tier": "synthetic",
            "n_play_calls": r.get("n_play_calls", 0),
            "domain": r.get("domain"), "teacher": r.get("teacher_model"),
        }))
    return out


def load_gold(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [_example(r["prompt"], r["code"], {**r.get("meta", {}), "tier": "gold"})
            for r in (json.loads(l) for l in path.open())]


def build(out_dir: Path, gated: Path, synthetic: Path, gold: Path,
          gold_weight: int = 6, synth_weight: int = 2,
          val_frac: float = 0.06, seed: int = 17) -> dict:
    rows: list[dict] = []
    for r in load_gated(gated):
        r["meta"].setdefault("tier", "gated")
        rows.append(r)
    rows += load_synthetic(synthetic)
    rows += load_gold(gold)

    # Group by prompt so paraphrases cannot straddle the split.
    groups: dict[str, list[dict]] = {}
    for r in rows:
        prompt = next(m["content"] for m in r["messages"] if m["role"] == "user")
        groups.setdefault(_key(prompt), []).append(r)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_val = max(1, int(len(keys) * val_frac))
    val_keys, train_keys = set(keys[:n_val]), keys[n_val:]

    def expand(ks) -> list[dict]:
        out = []
        for k in ks:
            for r in groups[k]:
                tier = r["meta"].get("tier", "gated")
                reps = {"gold": gold_weight, "synthetic": synth_weight}.get(tier, 1)
                # Static scenes teach the model to emit no animation; halve them.
                if r["meta"].get("n_play_calls", 1) == 0:
                    reps = max(1, reps // 2)
                out.extend([r] * reps)
        return out

    train, val = expand(train_keys), expand(val_keys)
    random.Random(seed).shuffle(train)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("valid", val)):
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for r in data:
                f.write(json.dumps({"messages": r["messages"]}) + "\n")

    tiers = Counter(r["meta"].get("tier", "gated") for r in rows)
    return {
        "unique_rows": len(rows),
        "by_tier": dict(tiers),
        "prompt_groups": len(groups),
        "train_examples": len(train),
        "valid_examples": len(val),
        "out": str(out_dir),
    }
