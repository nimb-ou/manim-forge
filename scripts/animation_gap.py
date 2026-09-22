#!/usr/bin/env python3
"""Does the animation measure separate the gold scenes from the corpus?

    ./.venv/bin/python scripts/animation_gap.py

Phase 2's exit condition, in one command. The measure in
forge/gate/animation.py is only worth anything if the 44 hand-written gold
scenes score visibly above a random sample of the training mix -- *without
squinting*. If they overlap, the measure is measuring nothing and rebuilding
the mix against it would be rebuilding it against noise.

So this reports the gap per component, not just overall: a measure that
separates on one signal and not the others is telling you which signal is
doing the work, and which parts of the vocabulary list are decoration.

It also reports the separation as an AUC -- the chance a random gold scene
outscores a random corpus row. 0.5 is a coin flip; 1.0 is perfect. That
number is the honest headline, because two means can differ while the
distributions sit on top of each other.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_measure():
    """Import the module without dragging in the whole forge package.

    `import forge.gate.animation` pulls forge/__init__ and manim behind it,
    which takes long enough on a busy machine to look like a hang. The
    module itself needs only ast and re, so it is loaded directly -- and
    registered in sys.modules first, because @dataclass looks its own module
    up by name while the class body is being processed.
    """
    spec = importlib.util.spec_from_file_location(
        "forge_animation", ROOT / "forge" / "gate" / "animation.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["forge_animation"] = mod
    spec.loader.exec_module(mod)
    return mod


def corpus_rows(limit: int, seed: int) -> list[tuple[str, str]]:
    """(id, code) from the training mix the adapter was actually trained on."""
    out = []
    path = ROOT / "kaggle" / "manim-forge-data" / "train.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # The gold scenes are *in* the training mix -- 240 of its 3010 rows --
        # so a random sample of it is about 8% gold, and the comparison was
        # quietly scoring part of the positive set as the negative one.
        if row.get("meta", {}).get("source") == "gold":
            continue
        code = row["messages"][-1]["content"]
        # strip a markdown fence if the assistant turn carries one
        if "```" in code:
            parts = code.split("```")
            code = max(parts, key=len)
            if code.startswith("python"):
                code = code[len("python"):]
        out.append((row.get("meta", {}).get("id", "?"), code))
    random.Random(seed).shuffle(out)
    return out[:limit]


def gold_rows() -> list[tuple[str, str]]:
    path = ROOT / "data" / "gold" / "gold.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [(r["meta"]["scene"], r["code"]) for r in rows]


def auc(pos: list[float], neg: list[float]) -> float:
    """P(a random positive outscores a random negative), ties at half."""
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0)
               for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="corpus rows to sample")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--show", type=int, default=0,
                    help="print this many corpus rows that score highest")
    a = ap.parse_args()

    m = load_measure()
    gold = [(n, m.profile(c)) for n, c in gold_rows()]
    corp = [(n, m.profile(c)) for n, c in corpus_rows(a.n, a.seed)]
    unparsed = sum(1 for _, p in corp if not p.parsed)
    gold = [(n, p) for n, p in gold if p.parsed]
    corp = [(n, p) for n, p in corp if p.parsed]

    print(f"{len(gold)} gold scenes vs {len(corp)} corpus rows "
          f"({unparsed} unparsed, dropped)\n")
    components = ["pacing", "grounding", "structure", "duration", "substance"]
    print(f"{'signal':12s} {'gold':>8s} {'corpus':>8s} {'gap':>8s} {'AUC':>7s}")
    print("-" * 47)
    for key in components:
        g = [p.scores[key] for _, p in gold]
        c = [p.scores[key] for _, p in corp]
        print(f"{key:12s} {mean(g):8.3f} {mean(c):8.3f} "
              f"{mean(g)-mean(c):+8.3f} {auc(g, c):7.3f}")
    g = [p.score for _, p in gold]
    c = [p.score for _, p in corp]
    print("-" * 47)
    print(f"{'COMPOSITE':12s} {mean(g):8.3f} {mean(c):8.3f} "
          f"{mean(g)-mean(c):+8.3f} {auc(g, c):7.3f}")

    print("\nraw habits (mean per scene):")
    raw = ["n_play", "n_sections", "continuous", "transforms", "discrete",
           "animate_calls", "text_objs", "geom_objs", "asserts",
           "computed_labels", "declared_seconds", "loc"]
    print(f"{'count':18s} {'gold':>8s} {'corpus':>8s}")
    for key in raw:
        gv = mean([getattr(p, key) for _, p in gold])
        cv = mean([getattr(p, key) for _, p in corp])
        print(f"  {key:16s} {gv:8.2f} {cv:8.2f}")

    # Which individual habits separate, before any weighting. A composite
    # can look strong while one component carries it and the rest add noise,
    # and the only way to see that is per-signal. Rates as well as counts,
    # because the gold scenes are 2.4x longer and a raw count rewards length
    # rather than habit.
    print("\nseparation of each raw habit (AUC; 0.5 is a coin flip):")
    per = {}
    for key in raw:
        per[key] = auc([float(getattr(p, key)) for _, p in gold],
                       [float(getattr(p, key)) for _, p in corp])
    for key in ("continuous", "animate_calls", "transforms", "discrete",
                "asserts", "computed_labels"):
        def rate_of(p, k=key):
            return float(getattr(p, k)) / max(p.loc, 1) * 100
        per[f"{key}/100loc"] = auc([rate_of(p) for _, p in gold],
                                   [rate_of(p) for _, p in corp])
        def per_play(p, k=key):
            return float(getattr(p, k)) / max(p.n_play, 1)
        per[f"{key}/play"] = auc([per_play(p) for _, p in gold],
                                 [per_play(p) for _, p in corp])
    for key, v in sorted(per.items(), key=lambda kv: -abs(kv[1] - 0.5)):
        mark = "  <-- separates" if abs(v - 0.5) > 0.2 else ""
        print(f"  {key:26s} {v:6.3f}{mark}")

    overlap = sum(1 for x in c if x >= min(g))
    print(f"\n{overlap} of {len(c)} corpus rows score at or above the *weakest* "
          f"gold scene ({100*overlap/len(c):.0f}%)")

    if a.show:
        print(f"\nhighest-scoring corpus rows — worth reading, they are either "
              f"genuinely good or the measure's blind spot:")
        for n, p in sorted(corp, key=lambda x: -x[1].score)[:a.show]:
            print(f"  {p.score:.3f}  {n}  {p.scores}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
