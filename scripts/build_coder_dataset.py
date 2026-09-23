#!/usr/bin/env python3
"""Assemble a train/valid split for Kaggle, for either adapter.

    ./.venv/bin/python scripts/build_coder_dataset.py                # coder
    ./.venv/bin/python scripts/build_coder_dataset.py --which planner

Both adapters need the same thing and the same guard, so they share this
rather than growing a second copy that drifts.

The coder's two sources, both beat->method:

  gold    182 rows, hand-written, the style target
  corpus  8,649 rows decomposed from render-verified scenes

The planner's one source is `plan_windows.jsonl`: incremental windows of the
190 arcs, grouped by arc for the same reason.

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
    ap.add_argument("--which", choices=("coder", "planner"), default="coder")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gold-weight", type=int, default=6)
    ap.add_argument("--valid-frac", type=float, default=0.04)
    a = ap.parse_args()

    if a.which == "coder":
        rows = load(ROOT / "data" / "planner" / "coder.jsonl") \
            + load(ROOT / "data" / "planner" / "coder_corpus.jsonl")
        out = a.out or ROOT / "kaggle" / "manim-forge-coder"
        # The grouping key is what must not straddle the split. For the coder
        # that is the scene; for the planner it is the arc, since one arc's
        # windows share its request and its running history.
        key = lambda r: r["meta"]["scene"]                     # noqa: E731
    else:
        rows = load(ROOT / "data" / "planner" / "plan_windows.jsonl")
        # Synthetic arcs join at a lower weight, by being loaded once while
        # the real ones are repeated. 146 real 3Blue1Brown arcs are the thing
        # being imitated; a teacher's imitation of them is worth less, and at
        # equal weight the next planner number would not say which taught it.
        rows += load(ROOT / "data" / "planner" / "plan_synth_windows.jsonl")
        out = a.out or ROOT / "kaggle" / "manim-forge-planner"
        key = lambda r: r["meta"]["id"].rsplit(":w", 1)[0]     # noqa: E731
    if not rows:
        print(f"no {a.which} rows on disk")
        return 1

    by_scene: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scene[key(r)].append(r)

    # Deterministic, and by scene. Hashing the scene id means the same scene
    # lands on the same side on every rebuild, so two runs stay comparable.
    valid_scenes = {
        s for s in by_scene
        if int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 1000
        < a.valid_frac * 1000}

    train, valid = [], []
    for scene, group in by_scene.items():
        target = valid if scene in valid_scenes else train
        src0 = group[0]["meta"]["source"]
        reps = a.gold_weight if src0 == "gold" else \
            (3 if src0 == "narration" else 1)
        # Gold is weighted in training only. Repeating it in validation would
        # make the eval loss a measurement of how well 44 scenes were
        # memorised.
        target.extend(group * (reps if target is train else 1))

    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("valid", valid)):
        (out / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in data) + "\n")

    mix = hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest()[:16]
    print(f"{len(train)} train / {len(valid)} valid rows "
          f"from {len(by_scene)} scenes ({len(valid_scenes)} held out)")
    src: dict[str, int] = defaultdict(int)
    for r in train:
        src[r["meta"]["source"]] += 1
    print("  train by source: " + ", ".join(f"{k} {v}" for k, v in
                                            sorted(src.items(), key=lambda x: -x[1])))
    print(f"  mix {mix} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
