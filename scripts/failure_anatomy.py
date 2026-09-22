#!/usr/bin/env python3
"""What kind of code does a run fail on, compared to what it passes on?

    ./.venv/bin/python scripts/failure_anatomy.py \
        --control data/bench/rounds4_n100_r4.json \
        --tuned   data/bench/tuned100_n100_r4.json

A render rate says how often a run worked. It does not say whether the
failures are small mistakes or a different kind of attempt, and those want
opposite responses.

Run 17 fell from 93% to 77%, and paired_eval.py attributed 75% of that to
repair no longer rescuing anything. This asks the next question -- *what was
repair being asked to rescue* -- by measuring the code itself:

    on the 18 prompts tuning lost
      control code (passed)   loc 16.4   calls 12.0   play 0.89
      tuned code (failed)     loc 23.8   calls 19.8   play 2.06

    tuned passes              loc 12.7   calls 11.0
    tuned failures            loc 22.9   calls 19.2

The tuned model's failures are 80% longer than its own passes and reach for
50% more play calls than the untuned model overall. Reading them confirms
it: where the control wrote `Dodecahedron()`, the tuned model hand-built a
`Polyhedron` from vertex coordinates; where the control drew a
`SurroundingRectangle`, it tried `table[0, :].set_color(...).animate`.

So the corpus did move the model in the intended direction -- more
animation, more construction -- and its API competence did not move with it.
Repair cannot rescue that, because the approach is wrong rather than the
line. That is a different problem from "repair regressed", and it is not
visible in any rate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_measure():
    spec = importlib.util.spec_from_file_location(
        "forge_animation", ROOT / "forge" / "gate" / "animation.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["forge_animation"] = mod
    spec.loader.exec_module(mod)
    return mod


def trials(path: Path) -> dict[int, dict]:
    raw = json.loads(path.read_text())
    rows = raw if isinstance(raw, list) else raw.get("trials", [])
    return {r.get("index", i): r for i, r in enumerate(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True, type=Path)
    ap.add_argument("--tuned", required=True, type=Path)
    a = ap.parse_args()
    m = load_measure()
    c, t = trials(a.control), trials(a.tuned)

    def row(label: str, codes: list[str]) -> None:
        ps = [p for p in (m.profile(x) for x in codes) if p.parsed]
        if not ps:
            print(f"  {label:30s} (nothing parsed)")
            return
        n = len(ps)
        print(f"  {label:30s} n={n:3d}  loc={sum(p.loc for p in ps)/n:6.1f}  "
              f"calls={sum(p.n_calls for p in ps)/n:6.1f}  "
              f"play={sum(p.n_play for p in ps)/n:5.2f}  "
              f"animate={sum(p.animate_calls for p in ps)/n:5.2f}  "
              f"anim-score={sum(p.score for p in ps)/n:.3f}")

    lost = [i for i in t if not t[i].get("ok") and c.get(i, {}).get("ok")]
    print(f"on the {len(lost)} prompts the tuned run lost:")
    row("control code (passed)", [c[i].get("code", "") for i in lost])
    row("tuned code (failed)", [t[i].get("code", "") for i in lost])

    print("\nacross every prompt:")
    row("control", [x.get("code", "") for x in c.values()])
    row("tuned", [x.get("code", "") for x in t.values()])

    print("\nwithin the tuned run:")
    row("its passes", [x.get("code", "") for x in t.values() if x.get("ok")])
    row("its failures", [x.get("code", "") for x in t.values()
                         if not x.get("ok")])

    print("\nIf the failures are markedly longer than the passes, the model is "
          "attempting\nmore than it can get right, and more repair rounds will "
          "not help -- the\napproach is wrong rather than the line.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
