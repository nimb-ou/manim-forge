"""Run the Phase 2 render gate over the normalised corpus."""
import argparse, json
from pathlib import Path
from forge.gate.run import gate_corpus

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="data/normalized/corpus.jsonl")
ap.add_argument("--out", default="data/verified/gate.jsonl")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--timeout", type=int, default=90)
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

s = gate_corpus(Path(a.corpus), Path(a.out), workers=a.workers,
                timeout=a.timeout, limit=a.limit)
print("\n" + "="*58)
print("  RENDER GATE")
print("="*58)
for k, v in s.items():
    print(f"{k:>16}: {v}")
print("="*58)
