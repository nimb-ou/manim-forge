#!/usr/bin/env python3
"""Compare benchmark runs prompt by prompt, not headline against headline.

    ./.venv/bin/python scripts/paired_eval.py \
        --control data/bench/ctrl100_n100_r4.json \
        --tuned   data/bench/tuned100_n100_r4.json

Two aggregate rates can differ for reasons that have nothing to do with the
change under test. The trials are the same prompts in the same order, so
they pair, and a paired view answers the question the aggregates cannot:

**when the tuned model fails a prompt the control passed, is it failing
differently, or failing the same way and not recovering?**

Each trial carries `history`: the error at every repair round, ending in
'none' if it was fixed. So history[0] is what the model produced *before any
repair* -- the generation -- and the rest is the repair loop. Splitting the
two is the whole point:

  same first error, control recovers, tuned does not  -> repair regressed
  different first error                               -> generation changed

Those call for different work, and the headline rate cannot tell them apart.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> dict[int, dict]:
    raw = json.loads(path.read_text())
    trials = raw if isinstance(raw, list) else raw.get("trials", [])
    return {t.get("index", i): t for i, t in enumerate(trials)}


def first_error(t: dict) -> str:
    h = t.get("history") or []
    return h[0] if h else ("none" if t.get("ok") else "unknown")


def rate(trials: dict[int, dict]) -> tuple[int, int, int]:
    ok = sum(1 for t in trials.values() if t.get("ok"))
    first = sum(1 for t in trials.values()
                if t.get("ok") and t.get("rounds", 0) == 0)
    return ok, first, ok - first


def describe(name: str, trials: dict[int, dict]) -> None:
    ok, first, rescued = rate(trials)
    n = len(trials)
    print(f"{name:10s} n={n:3d}  pass={ok:3d} ({100*ok/n:.0f}%)  "
          f"first-try={first:3d} ({100*first/n:.0f}%)  "
          f"rescued-by-repair={rescued}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True, type=Path)
    ap.add_argument("--tuned", required=True, type=Path)
    ap.add_argument("--show", type=int, default=6,
                    help="how many disagreeing prompts to print in full")
    a = ap.parse_args()

    c, t = load(a.control), load(a.tuned)
    shared = sorted(set(c) & set(t))
    if not shared:
        print("no shared prompt indices — these are not the same benchmark")
        return 1
    if len(shared) != len(c) or len(shared) != len(t):
        print(f"note: comparing {len(shared)} shared of "
              f"{len(c)} control / {len(t)} tuned")
    c = {i: c[i] for i in shared}
    t = {i: t[i] for i in shared}

    print("=" * 74)
    describe("control", c)
    describe("tuned", t)

    both = [i for i in shared if c[i].get("ok") and t[i].get("ok")]
    neither = [i for i in shared if not c[i].get("ok") and not t[i].get("ok")]
    lost = [i for i in shared if c[i].get("ok") and not t[i].get("ok")]
    won = [i for i in shared if not c[i].get("ok") and t[i].get("ok")]
    print(f"\nboth pass {len(both)} | both fail {len(neither)} | "
          f"tuning LOST {len(lost)} | tuning WON {len(won)}")

    # ── the question ──────────────────────────────────────────────────────
    # Decomposed, not labelled. The first version of this printed one of two
    # verdicts based on whether the lost prompts shared a first-round error
    # with the control -- and got it backwards on the real data, because on
    # 12 of 18 lost prompts the control passed *first try*, so there was no
    # first-round error to share and absence read as difference. The two
    # mechanisms are not exclusive anyway: a run can generate slightly worse
    # code *and* stop repairing it, which is what happened.
    c_ok, c_first, c_resc = rate(c)
    t_ok, t_first, t_resc = rate(t)
    print(f"\nwhere the {c_ok - t_ok:+d} went:")
    print(f"  generation : first-try {c_first} -> {t_first} "
          f"({t_first - c_first:+d})")
    print(f"  repair     : rescued   {c_resc} -> {t_resc} "
          f"({t_resc - c_resc:+d})")
    if (c_ok - t_ok) != 0:
        share = abs(t_resc - c_resc) / max(abs(c_ok - t_ok), 1)
        print(f"  -> repair accounts for {100*share:.0f}% of the difference")
    if lost:
        same_first = [i for i in lost if first_error(c[i]) == first_error(t[i])]
        c_rescued = [i for i in lost if c[i].get("rounds", 0) > 0]
        print(f"  of the {len(lost)} prompts lost: {len(same_first)} share the "
              f"control's first-round error, {len(c_rescued)} were ones the "
              f"control only passed because repair fixed them")

    print("\nfirst-round error mix (what the model writes before any repair):")
    fc, ft = Counter(first_error(x) for x in c.values()), \
        Counter(first_error(x) for x in t.values())
    for key in sorted(set(fc) | set(ft), key=lambda k: -(fc[k] + ft[k])):
        print(f"  {key:16s} control {fc[key]:3d}   tuned {ft[key]:3d}")

    print("\nrepair effectiveness (of trials that needed at least one round):")
    for name, d in (("control", c), ("tuned", t)):
        needed = [x for x in d.values() if (x.get("history") or [None])[0]
                  not in (None, "none")]
        fixed = [x for x in needed if x.get("ok")]
        pct = f"{100*len(fixed)/len(needed):.0f}%" if needed else "n/a"
        print(f"  {name:8s} {len(fixed):3d} of {len(needed):3d} recovered "
              f"({pct})")

    if lost and a.show:
        print(f"\n{'=' * 74}\nprompts tuning lost (first {a.show}):")
        for i in lost[:a.show]:
            print(f"\n  [{i}] control: ok rounds={c[i].get('rounds')} "
                  f"history={c[i].get('history')}")
            print(f"       tuned:  FAIL rounds={t[i].get('rounds')} "
                  f"history={t[i].get('history')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
