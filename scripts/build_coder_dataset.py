#!/usr/bin/env python3
"""Assemble the coder's train/valid split for Kaggle.

    ./.venv/bin/python scripts/build_coder_dataset.py

Two sources, both beat->method:

  gold    182 rows, hand-written, the style target
  corpus  8,649 rows decomposed from render-verified scenes

Split **by scene**, not by row. Beats from one scene share its request, its
objects and its helpers, so a random row split would put beat 3 in training
and beat 4 in validation and the eval number would be memorisation. This
project has already published one number that measured the wrong thing; the
cheap guard is to never let a scene straddle the split.

Gold is repeated, because 182 rows against 8,649 would otherwise be 2% of
the mix and the style target would vanish into the corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "kaggle" / "manim-forge-coder")
    ap.add_argument("--gold-weight", type=int, default=6)
    ap.add_argument("--valid-frac", type=float, default=0.04)
    a = ap.parse_args()

    rows = load(ROOT / "data" / "planner" / "coder.jsonl") \
        + load(ROOT / "data" / "planner" / "coder_corpus.jsonl")
    if not rows:
        print("no coder rows — run build_coder_mix.py and decompose_corpus.py")
        return 1

    by_scene: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scene[r["meta"]["scene"]].append(r)

    # Deterministic, and by scene. Hashing the scene id means the same scene
    # lands on the same side on every rebuild, so two runs stay comparable.
    valid_scenes = {
        s for s in by_scene
        if int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 1000
        < a.valid_frac * 1000}

    train, valid = [], []
    for scene, group in by_scene.items():
        target = valid if scene in valid_scenes else train
        reps = a.gold_weight if group[0]["meta"]["source"] == "gold" else 1
        # Gold is weighted in training only. Repeating it in validation would
        # make the eval loss a measurement of how well 44 scenes were
        # memorised.
        target.extend(group * (reps if target is train else 1))

    a.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("valid", valid)):
        (a.out / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in data) + "\n")

    mix = hashlib.sha256((a.out / "train.jsonl").read_bytes()).hexdigest()[:16]
    print(f"{len(train)} train / {len(valid)} valid rows "
          f"from {len(by_scene)} scenes ({len(valid_scenes)} held out)")
    src: dict[str, int] = defaultdict(int)
    for r in train:
        src[r["meta"]["source"]] += 1
    print("  train by source: " + ", ".join(f"{k} {v}" for k, v in
                                            sorted(src.items(), key=lambda x: -x[1])))
    print(f"  mix {mix} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
