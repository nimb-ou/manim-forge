"""Phase 1 — pull every source into one JSONL corpus."""
import json, sys
from collections import Counter
from pathlib import Path

from forge.ingest.sources import SOURCES

out = Path("data/normalized/corpus.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

rows, seen, dupes = [], {}, 0
for name, fn in SOURCES.items():
    print(f"-> {name} ...", end=" ", flush=True)
    try:
        got = list(fn())
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        continue
    kept = 0
    for r in got:
        if r.dedupe_key in seen:
            dupes += 1
            continue
        seen[r.dedupe_key] = r.id
        rows.append(r); kept += 1
    print(f"{len(got)} rows, {kept} new, {len(got)-kept} dupes")

with out.open("w") as f:
    for r in rows:
        f.write(json.dumps(r.to_dict()) + "\n")

print(f"\n{'='*62}\nCORPUS: {len(rows)} unique rows  ({dupes} duplicates removed)\n{'='*62}")

def tally(label, counter, total):
    print(f"\n{label}")
    for k, v in counter.most_common():
        print(f"  {str(k):<26} {v:>6}  {v/total*100:5.1f}%")

n = len(rows)
tally("dialect", Counter(r.flavor for r in rows), n)
tally("licence", Counter(r.license for r in rows), n)
tally("source",  Counter(r.source for r in rows), n)

animated = sum(r.is_animated for r in rows)
parses   = sum(r.parses for r in rows)
print(f"\nanimated (calls .play)      {animated:>6}  {animated/n*100:5.1f}%")
print(f"static  (never animates)    {n-animated:>6}  {(n-animated)/n*100:5.1f}%")
print(f"parses as valid Python      {parses:>6}  {parses/n*100:5.1f}%")
pc = [r.n_play_calls for r in rows if r.is_animated]
if pc:
    pc.sort()
    print(f"\nplay() calls among animated: median {pc[len(pc)//2]}, max {pc[-1]}")
print(f"\nwritten to {out}  ({out.stat().st_size/1e6:.1f} MB)")
