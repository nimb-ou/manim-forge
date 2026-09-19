import argparse, json
from pathlib import Path
from forge.gate.export import build

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/normalized/corpus.jsonl")
    ap.add_argument("--gate", default="data/verified/gate.jsonl")
    ap.add_argument("--out", default="data/verified/train.jsonl")
    ap.add_argument("--animated-weight", type=int, default=2)
    a = ap.parse_args()
    s = build(Path(a.corpus), Path(a.gate), Path(a.out), a.animated_weight)
    print("="*58); print("  VERIFIED TRAINING SET"); print("="*58)
    for k, v in s.items(): print(f"{k:>34}: {v}")
    print("="*58)

if __name__ == "__main__":
    main()
