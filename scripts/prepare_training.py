import argparse
from pathlib import Path
from forge.train.prepare import build

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/train")
    ap.add_argument("--gated", default="data/verified/train.jsonl")
    ap.add_argument("--synthetic", default="data/synthetic/generated.jsonl")
    ap.add_argument("--gold", default="data/gold/gold.jsonl")
    ap.add_argument("--gold-weight", type=int, default=6)
    ap.add_argument("--synth-weight", type=int, default=2)
    a = ap.parse_args()
    s = build(Path(a.out), Path(a.gated), Path(a.synthetic), Path(a.gold),
              a.gold_weight, a.synth_weight)
    print("="*56); print("  TRAINING MIX"); print("="*56)
    for k, v in s.items(): print(f"{k:>18}: {v}")
    print("="*56)

if __name__ == "__main__":
    main()
