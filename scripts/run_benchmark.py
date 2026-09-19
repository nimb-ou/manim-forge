"""Run the baseline benchmark on ManimBench's held-out test split."""
import argparse, json
from pathlib import Path

import pandas as pd
from forge.harness import RenderHarness
from forge.evaluate.benchmark import Benchmark

TEST_URL = ("https://huggingface.co/datasets/SuienR/ManimBench-v1/resolve/"
            "refs%2Fconvert%2Fparquet/default/test/0000.parquet")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=20, help="prompts to run (100 = full split)")
ap.add_argument("--out", default=None)
a = ap.parse_args()

df = pd.read_parquet(TEST_URL)
# 'Reviewed Description' is the populated column; 'Generated Description' is
# 94% empty despite what the dataset card says.
prompts = df["Reviewed Description"].dropna().tolist()[: a.n]
print(f"{len(prompts)} prompts from the held-out test split\n")

h = RenderHarness(python_bin="./.venv/bin/python", cache_dir="data/frames", timeout=120)
out = Path(a.out or f"data/bench/{a.model.split('/')[-1]}_n{a.n}.json")
summary = Benchmark(h).run_model(a.model, prompts, out)

print("\n" + "=" * 58)
for k, v in summary.items():
    print(f"{k:>22}: {v}")
print("=" * 58)
print(f"written to {out}")
