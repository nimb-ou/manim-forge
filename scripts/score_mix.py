#!/usr/bin/env python3
"""Score every row of the training mix with the animation gate.

    ./.venv/bin/python scripts/score_mix.py --out data/mix_scores.jsonl

Phase 2's second half. The measure separates gold from corpus at AUC 0.98;
this asks what the mix actually looks like through it, and what would be
left at each threshold -- because "rebuild the mix against the measure" is a
sentence that hides a decision about how much corpus to throw away.

Reports by source, because the mix is not one thing: scraped rows, synthetic
rows from a teacher, bespoke rows and the gold scenes have different
problems and a single threshold treats them identically.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIX = ROOT / "kaggle" / "manim-forge-data" / "train.jsonl"


def load_measure():
    spec = importlib.util.spec_from_file_location(
        "forge_animation", ROOT / "forge" / "gate" / "animation.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["forge_animation"] = mod
    spec.loader.exec_module(mod)
    return mod


def code_of(row: dict) -> str:
    code = row["messages"][-1]["content"]
    if "```" in code:
        code = max(code.split("```"), key=len)
        if code.startswith("python"):
            code = code[len("python"):]
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "mix_scores.jsonl")
    ap.add_argument("--mix", type=Path, default=MIX)
    a = ap.parse_args()

    m = load_measure()
    rows = [json.loads(l) for l in a.mix.read_text().splitlines() if l.strip()]
    by_source: dict[str, list[float]] = defaultdict(list)
    out_lines, unparsed = [], 0

    for row in rows:
        meta = row.get("meta", {})
        p = m.profile(code_of(row))
        if not p.parsed:
            unparsed += 1
        src = meta.get("source", "?")
        by_source[src].append(p.score)
        out_lines.append(json.dumps({
            "id": meta.get("id"), "source": src, "score": p.score,
            "convention": p.convention, "components": p.scores,
            "n_play": p.n_play, "loc": p.loc, "parsed": p.parsed}))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(out_lines) + "\n")
    print(f"{len(rows)} rows scored -> {a.out}  ({unparsed} unparsed)\n")

    print(f"{'source':18s} {'rows':>6s} {'mean':>7s} {'median':>7s} {'p90':>7s}")
    print("-" * 50)
    allsc = []
    for src in sorted(by_source, key=lambda s: -len(by_source[s])):
        v = sorted(by_source[src])
        allsc += v
        med = v[len(v) // 2]
        p90 = v[int(0.9 * (len(v) - 1))]
        print(f"{src:18s} {len(v):6d} {sum(v)/len(v):7.3f} {med:7.3f} {p90:7.3f}")
    allsc.sort()
    print("-" * 50)
    print(f"{'ALL':18s} {len(allsc):6d} {sum(allsc)/len(allsc):7.3f} "
          f"{allsc[len(allsc)//2]:7.3f} {allsc[int(0.9*(len(allsc)-1))]:7.3f}")

    print("\nwhat a threshold would keep:")
    print(f"{'cut':>6s} {'kept':>7s} {'share':>7s}   by source")
    for cut in (0.30, 0.40, 0.50, 0.60, 0.70):
        kept = [s for s in allsc if s >= cut]
        detail = "  ".join(
            f"{src}:{sum(1 for x in by_source[src] if x >= cut)}"
            for src in sorted(by_source, key=lambda s: -len(by_source[s])))
        print(f"{cut:6.2f} {len(kept):7d} {100*len(kept)/len(allsc):6.1f}%   {detail}")

    print("\nA threshold is not free: every row cut is a row the model no "
          "longer sees.\nThe gold scenes average 0.73, so a cut that keeps "
          "the mix large keeps\nmost of what the measure calls slideware, and "
          "a cut that matches gold\nleaves a corpus too small to train on. "
          "That trade is the decision, and\nit should be made from this table "
          "rather than from a round number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
