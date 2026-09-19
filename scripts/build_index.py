import time
from pathlib import Path
from forge.retrieve.examples import ExampleIndex

t = time.monotonic()
idx = ExampleIndex.build(Path("data/verified/train.jsonl"))
idx.save(Path("data/verified/example_index.jsonl"))
print(f"indexed {len(idx.examples)} verified scenes in {time.monotonic()-t:.0f}s")

for q in ["explain eigenvectors of a matrix", "show a chessboard with pieces"]:
    print(f"\nquery: {q!r}")
    for score, ex in idx.search(q, k=3):
        print(f"  {score:.3f}  [{ex.source}] {ex.prompt[:66]}")
