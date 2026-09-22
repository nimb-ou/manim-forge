"""Compare an eval run against its baseline, and say what it means.

Phase 1 exists to answer one question, and it is not "did the render rate go
up". The untuned model already renders 93% of single scenes; a fine-tune
nudging that to 95% would prove almost nothing, because rendering was never
the thing that was broken.

The question is whether the model stops answering a *smaller* question than
the one asked. Against 81 real 3Blue1Brown titles the untuned model returns
16.3 seconds where the real video runs 16 minutes -- a length ratio of
1.76% -- and matches 8.8% of the concepts those videos spend their time on.
docs/PLAN.md §2 predicts a fine-tune on the current corpus will not move
either, because the corpus itself is 1.3% ValueTracker against gold's 9.1%.

So this prints render rate, and then tells you to look at the other two.

    python scripts/compare_eval.py \\
        --bench data/bench/tuned100_n100_r4.json \\
        --hard  data/bench/tuned81_n81.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

BASE_BENCH = Path("data/bench/rounds4_n100_r4.json")
BASE_HARD = Path("data/bench/hard_r4_n81.json")


def trials(path: Path) -> list[dict]:
    """Accept both shapes: a bare list, and {meta, trials}."""
    d = json.loads(path.read_text())
    return d["trials"] if isinstance(d, dict) else d


def meta(path: Path) -> dict:
    d = json.loads(path.read_text())
    return d.get("meta", {}) if isinstance(d, dict) else {}


def bench_stats(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("ok")]
    return {
        "render success": len(ok) / len(rows),
        "first try": sum(1 for r in ok if r.get("rounds") == 0) / len(rows),
        "rescued by repair": sum(1 for r in ok if r.get("rounds", 0) > 0) / len(rows),
    }


def hard_stats(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return {"render success": 0.0}
    ratios = [r["seconds"] / r["real_seconds"] for r in ok
              if r.get("real_seconds")]
    return {
        "render success": len(ok) / len(rows),
        "mean duration s": st.mean(r["seconds"] for r in ok),
        "length ratio": st.mean(ratios) if ratios else 0.0,
        "concept coverage": st.mean(r.get("coverage", 0) for r in ok),
        # Coverage is computed from the *code*, so a scene that failed to
        # render still has one. Averaging over renders only is consistent
        # between runs and quietly selects for whichever scenes happened to
        # work -- which matters here, because run 17's failures are its more
        # ambitious attempts, so the rendered-only figure under-reports
        # exactly the thing being measured. Both are printed; neither is
        # the honest one on its own.
        "concept coverage (all)": st.mean(r.get("coverage", 0) for r in rows),
        "mean play calls": st.mean(r.get("n_play_calls", 0) for r in ok),
        "mean beats": st.mean(r.get("n_beats", 0) for r in ok),
    }


def table(title: str, base: dict, new: dict, pct_keys: set[str]) -> None:
    print(f"\n{title}")
    print("-" * 66)
    print(f"  {'metric':<20} {'baseline':>12} {'tuned':>12} {'change':>14}")
    for k in base:
        b, n = base[k], new.get(k, 0.0)
        if k in pct_keys:
            bs, ns = f"{b:.1%}", f"{n:.1%}"
            delta = f"{(n - b) * 100:+.1f} pts"
        else:
            bs, ns = f"{b:.2f}", f"{n:.2f}"
            delta = f"{n - b:+.2f}"
        print(f"  {k:<20} {bs:>12} {ns:>12} {delta:>14}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path)
    ap.add_argument("--hard", type=Path)
    ap.add_argument("--base-bench", type=Path, default=BASE_BENCH)
    ap.add_argument("--base-hard", type=Path, default=BASE_HARD)
    a = ap.parse_args()

    if a.bench and a.bench.exists():
        m = meta(a.bench)
        if m.get("adapter"):
            print(f"adapter: {m['adapter']}")
        if m.get("training"):
            t = m["training"]
            print(f"trained on mix {t.get('mix_sha256')} — "
                  f"{t.get('train_rows')} rows, {t.get('epochs')} epochs, "
                  f"LoRA r={t.get('lora_r')}")
        table("SINGLE-SCENE BENCHMARK (ManimBench held-out, repair rounds=4)",
              bench_stats(trials(a.base_bench)), bench_stats(trials(a.bench)),
              {"render success", "first try", "rescued by repair"})

    if a.hard and a.hard.exists():
        base, new = hard_stats(trials(a.base_hard)), hard_stats(trials(a.hard))
        table("HARD EVAL (81 real 3Blue1Brown titles)", base, new,
              {"render success", "length ratio", "concept coverage"})

        print("\n" + "=" * 66)
        print("  THE VERDICT")
        print("=" * 66)
        dl = (new["length ratio"] - base["length ratio"]) * 100
        dca = (new.get("concept coverage (all)", 0)
               - base.get("concept coverage (all)", 0)) * 100
        print(f"  concept coverage {dca:+.2f} pts over ALL trials   "
              f"({base.get('concept coverage (all)', 0):.1%} -> "
              f"{new.get('concept coverage (all)', 0):.1%})")
        dc = (new["concept coverage"] - base["concept coverage"]) * 100
        print(f"  length ratio     {dl:+.2f} pts   "
              f"({base['length ratio']:.2%} -> {new['length ratio']:.2%})")
        print(f"  concept coverage {dc:+.2f} pts   "
              f"({base['concept coverage']:.1%} -> {new['concept coverage']:.1%})")
        print()
        if dl < 0.5 and dc < 1.0:
            print("  Neither moved. This is what docs/PLAN.md §2 predicted: the")
            print("  corpus teaches slideware, so training on it harder teaches")
            print("  slideware harder. The render gate filters for *executes*")
            print("  and nothing filters for *animates*.")
            print()
            print("  That makes Phase 2 the whole project rather than a hunch,")
            print("  and this run the evidence for it. Not a wasted week.")
        else:
            print("  Something moved. Before believing it, check that only one")
            print("  variable changed between these runs -- that rule has been")
            print("  broken three times on this project already.")
        print("=" * 66)


if __name__ == "__main__":
    main()
