"""The one-shot model's dataset: request + nearest scenes in, whole scene out.

    ./.venv/bin/python scripts/build_oneshot_dataset.py
      -> kaggle/manim-forge-oneshot/{train,valid}.jsonl

Retrieval-augmented fine-tuning: every hand-written scene in the library
(forge/kit/library.py, held-out topics already removed) becomes a target,
shown two *other* scenes as examples, exactly as forge/app/oneshot.py
prompts at inference. Each scene appears twice:

  * with its two nearest neighbours -- the in-scope case, where a close
    example exists and the job is to adapt it (new numbers, new context);
  * with two neighbours from a random point further down the list -- the
    case where the examples are only loosely related and the model has to
    write more of the scene itself.

A scene is never shown its own request, and neighbours sharing its request
under different wording (cosine >= 0.9) are skipped, so no row is answered
by copying its own answer. Split by scene, 5% held out.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forge.app.oneshot import ONESHOT_SYSTEM, user_prompt  # noqa: E402
from forge.evaluate.heldout_guard import touches_heldout  # noqa: E402
from forge.kit.library import as_body, scenes, similar  # noqa: E402

OUT = ROOT / "kaggle" / "manim-forge-oneshot"
MAX_TOKENS = 3000


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    rng = random.Random(7)
    train, valid, lens = [], [], []
    too_long = 0
    for s in scenes():
        text = s["request"] + " " + " ".join(b["intent"] for b in s["beats"])
        assert not touches_heldout(text)
        near = [x for c, x in similar(s["request"] + " " + text, 12,
                                      exclude={s["request"]}) if c < 0.9]
        if len(near) < 4:
            continue
        lo = rng.randint(2, len(near) - 2)
        held = int(hashlib.sha256(s["request"].encode()).hexdigest(), 16) % 20 == 0
        for ex in (near[:2], near[lo: lo + 2]):
            msgs = [{"role": "system", "content": ONESHOT_SYSTEM},
                    {"role": "user", "content": user_prompt(s["request"], examples=ex)},
                    {"role": "assistant", "content": as_body(s)}]
            n = len(tok(tok.apply_chat_template(msgs, tokenize=False))["input_ids"])
            if n > MAX_TOKENS:
                too_long += 1
                continue
            lens.append(n)
            (valid if held else train).append(
                {"messages": msgs, "meta": {"request": s["request"],
                                            "source": s["source"]}})
    rng.shuffle(train)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("valid", valid)):
        (OUT / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part))
    meta = OUT / "dataset-metadata.json"
    if not meta.exists():
        meta.write_text(json.dumps({"title": "manim-forge-oneshot",
                                    "id": "nimbou/manim-forge-oneshot",
                                    "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}))
    lens.sort()
    print(f"{len(train)} train / {len(valid)} valid -> {OUT}; tokens median "
          f"{lens[len(lens) // 2]}, p90 {lens[int(.9 * len(lens))]}, max {lens[-1]}"
          f" ({too_long} over {MAX_TOKENS} dropped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
