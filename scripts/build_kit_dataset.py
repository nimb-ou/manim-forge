"""The kit coder's dataset for Kaggle, with the short trained prompt.

    ./.venv/bin/python scripts/build_kit_dataset.py
      -> kaggle/manim-forge-kit/{train,valid}.jsonl

Rows come from data/kit/kit_beats_clean.jsonl (teacher kit beats from
scenes that rendered, filtered for relevance and novelty). The teacher's
long API prompt is replaced by CODE_SYSTEM_KIT_TRAINED, which is what the
pipeline sends a trained kit coder. Split by scene, 5% held out.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forge.app.pipeline import CODE_SYSTEM_KIT_TRAINED  # noqa: E402

SRC = ROOT / "data" / "kit" / "kit_beats_clean.jsonl"
OUT = ROOT / "kaggle" / "manim-forge-kit"


def main() -> int:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "filter_kit_beats.py")],
                   check=True)
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    # The kernel trains at 1,024 tokens (median row 447, p90 566); the few
    # rows past it would lose their completion to truncation.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

    def n_tokens(msgs):
        return len(tok(tok.apply_chat_template(msgs, tokenize=False))["input_ids"])
    train, valid = [], []
    dropped = 0
    for r in rows:
        msgs = [{"role": "system", "content": CODE_SYSTEM_KIT_TRAINED},
                r["messages"][1], r["messages"][2]]
        if n_tokens(msgs) > 1000:
            dropped += 1
            continue
        meta = {k: str(v) for k, v in r["meta"].items()}
        scene = r["meta"]["scene"]
        held = int(hashlib.sha256(scene.encode()).hexdigest(), 16) % 20 == 0
        (valid if held else train).append({"messages": msgs, "meta": meta})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("valid", valid)):
        (OUT / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part))
    meta = OUT / "dataset-metadata.json"
    if not meta.exists():
        meta.write_text(json.dumps({"title": "manim-forge-kit",
                                    "id": "nimbou/manim-forge-kit",
                                    "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}))
    print(f"{len(train)} train / {len(valid)} valid -> {OUT} "
          f"({dropped} over 1,000 tokens dropped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
