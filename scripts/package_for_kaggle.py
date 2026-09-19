"""Package the corpus and code for upload as a Kaggle dataset.

Kaggle notebooks get a GPU and root, so LaTeX and ffmpeg install fine and the
render gate runs there too. That matters more than the GPU: GRPO's reward *is*
a render, so training and verification have to live in the same place.

What ships is the training data plus the forge package itself — the notebook
imports the same harness, lint and repair code that runs locally, so a reward
computed on Kaggle means the same thing as a score measured here.
"""
from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

OUT = Path("kaggle/manim-forge-data")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # 1. training splits
    for name in ("train", "valid"):
        src = Path(f"data/train/{name}.jsonl")
        if src.exists():
            shutil.copy2(src, OUT / f"{name}.jsonl")

    # 2. the example index, so retrieval works identically there
    idx = Path("data/verified/example_index.jsonl")
    if idx.exists():
        shutil.copy2(idx, OUT / "example_index.jsonl")

    # 3. held-out benchmark prompts, so eval is comparable across machines
    bench = Path("data/verified/bench_prompts.json")
    if not bench.exists():
        import pandas as pd
        df = pd.read_parquet(
            "https://huggingface.co/datasets/SuienR/ManimBench-v1/resolve/"
            "refs%2Fconvert%2Fparquet/default/test/0000.parquet")
        bench.parent.mkdir(parents=True, exist_ok=True)
        bench.write_text(json.dumps(
            {"prompts": df["Reviewed Description"].dropna().tolist()}, indent=2))
    shutil.copy2(bench, OUT / "bench_prompts.json")

    # 4. the forge package — same harness on both machines, so a reward
    #    computed there is the number measured here
    with tarfile.open(OUT / "forge.tar.gz", "w:gz") as tar:
        tar.add("forge", arcname="forge",
                filter=lambda t: None if "__pycache__" in t.name else t)

    sizes = {p.name: f"{p.stat().st_size/1e6:.1f} MB" for p in sorted(OUT.iterdir())}
    print(f"packaged -> {OUT}")
    for k, v in sizes.items():
        print(f"  {k:<24} {v}")
    total = sum(p.stat().st_size for p in OUT.iterdir()) / 1e6
    print(f"  {'TOTAL':<24} {total:.1f} MB")


if __name__ == "__main__":
    main()
