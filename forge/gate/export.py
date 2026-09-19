"""Turn gate results into a training-ready dataset.

The gate writes a verdict per row; the corpus holds the content. This joins
them and emits only what survived, in the chat format MLX LoRA expects.

Two decisions are made here rather than at training time, because both need
information only the gate has:

**Repaired code wins.** Where the lint fixed a row (a trailing self.wait() on a
static scene), the training target is the *repaired* source — that is the
version that actually renders, and teaching the model the broken original
would be teaching it the failure we just removed.

**Animated rows are weighted up.** The corpus is heavily static, and a model
trained on it writes scenes that produce no video. Duplicating animated rows is
the crudest possible fix and it is the right one to start with: it is
transparent, trivially tunable, and we can measure whether it helped.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

SYSTEM = (
    "You are an expert Manim Community Edition developer. "
    "Given a description of an animation, write complete, runnable Python code "
    "using `from manim import *` and a single Scene subclass. "
    "Always animate with self.play(...) and end with self.wait(). "
    "Output only code in one ``` block. No explanation."
)


def build(corpus: Path, gate: Path, out: Path,
          animated_weight: int = 2, min_prompt_chars: int = 20) -> dict:
    rows = {json.loads(l)["id"]: json.loads(l) for l in corpus.open()}

    verdicts: dict[str, dict] = {}
    for line in gate.open():
        try:
            v = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not v.get("is_env_failure"):
            verdicts[v["id"]] = v

    kept, stats = [], Counter()
    for rid, v in verdicts.items():
        if not v.get("ok"):
            stats["failed_gate"] += 1
            continue
        row = rows.get(rid)
        if row is None:
            stats["missing_source"] += 1
            continue
        prompt = (row.get("prompt") or "").strip()
        if len(prompt) < min_prompt_chars:
            stats["prompt_too_short"] += 1
            continue

        # The gate stores the repaired source only when linting changed it.
        code = v.get("code") or row["code"]

        rec = {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"```python\n{code}\n```"},
            ],
            "meta": {
                "id": rid, "source": row["source"], "license": row["license"],
                "n_play_calls": row["n_play_calls"], "flavor": row["flavor"],
                "animated": row["n_play_calls"] > 0,
                "video_s": v.get("duration_s"),
            },
        }
        reps = animated_weight if row["n_play_calls"] > 0 else 1
        kept.extend([rec] * reps)
        stats["kept"] += 1
        stats["animated" if row["n_play_calls"] > 0 else "static"] += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    return {
        "unique_verified": stats["kept"],
        "animated": stats["animated"],
        "static": stats["static"],
        "training_examples_after_weighting": len(kept),
        "dropped_failed_gate": stats["failed_gate"],
        "dropped_short_prompt": stats["prompt_too_short"],
        "out": str(out),
    }
