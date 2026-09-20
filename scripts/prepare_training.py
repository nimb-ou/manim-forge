import argparse
from pathlib import Path
from forge.train.prepare import build

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/train")
    ap.add_argument("--gated", default="data/verified/train.jsonl")
    ap.add_argument("--synthetic", action="append", default=None,
                    help="repeatable; defaults to all three synthetic files")
    ap.add_argument("--gold", default="data/gold/gold.jsonl")
    ap.add_argument("--gold-weight", type=int, default=6)
    ap.add_argument("--synth-weight", type=int, default=2)
    ap.add_argument("--keep-static", action="store_true",
                    help="keep rows with no self.play() call (default: drop)")
    a = ap.parse_args()
    synth = [Path(x) for x in (a.synthetic or [
        "data/synthetic/generated.jsonl",
        "data/synthetic/stream.jsonl",
        "data/synthetic/from_narration.jsonl",
    ])]
    s = build(Path(a.out), Path(a.gated), synth, Path(a.gold),
              a.gold_weight, a.synth_weight, drop_static=not a.keep_static)
    print("="*56); print("  TRAINING MIX"); print("="*56)
    for k, v in s.items(): print(f"{k:>18}: {v}")
    print("="*56)

if __name__ == "__main__":
    main()
