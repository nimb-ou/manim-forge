"""Re-measure the baseline with the repair loop enabled."""
import argparse, json, time
from pathlib import Path
from collections import Counter

import pandas as pd
from forge.harness import RenderHarness
from forge.repair.loop import RepairLoop

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--retrieval", action="store_true", help="add few-shot examples")
ap.add_argument("--tag", default="", help="label for the output file")
ap.add_argument("--adapter", default=None,
                help="MLX LoRA adapter directory. A PEFT adapter from Kaggle "
                     "must be converted first: scripts/peft_to_mlx.py")
a = ap.parse_args()

df = pd.read_parquet("https://huggingface.co/datasets/SuienR/ManimBench-v1/resolve/"
                     "refs%2Fconvert%2Fparquet/default/test/0000.parquet")
prompts = df["Reviewed Description"].dropna().tolist()[: a.n]

from mlx_lm import load
print(f"loading {a.model} ...", flush=True)
from forge.adapters import check_mlx_adapter
check_mlx_adapter(a.adapter)
if a.adapter:
    print(f"  adapter: {a.adapter}", flush=True)
model, tok = load(a.model, **({"adapter_path": a.adapter} if a.adapter else {}))
h = RenderHarness(python_bin="./.venv/bin/python", cache_dir="data/frames", timeout=120)
index = None
if a.retrieval:
    from forge.retrieve.examples import ExampleIndex
    index = ExampleIndex.load(Path("data/verified/example_index.jsonl"))
    print(f"retrieval on: {len(index.examples)} verified examples indexed")
loop = RepairLoop(model, tok, h, max_rounds=a.rounds, index=index)

run_meta = {"model": a.model, "adapter": a.adapter, "rounds": a.rounds,
            "retrieval": bool(a.retrieval), "n": a.n}
if a.adapter:
    rj = Path(a.adapter) / "run.json"
    if rj.exists():
        run_meta["training"] = json.loads(rj.read_text())

rows, t0 = [], time.monotonic()
for i, p in enumerate(prompts):
    t = time.monotonic()
    r = loop.run(p)
    rows.append({"index": i, "ok": r.ok, "rounds": r.rounds_used,
                 "final_error": r.final_error, "lint": r.lint_rules,
                 "history": r.history, "seconds": round(time.monotonic()-t, 1),
                 "code": r.code})
    print(f"  [{i+1:>3}/{len(prompts)}] {'PASS' if r.ok else 'FAIL':<5} "
          f"rounds={r.rounds_used} lint={r.lint_rules} {r.history}", flush=True)

n_ok = sum(r["ok"] for r in rows)
first_try = sum(1 for r in rows if r["ok"] and r["rounds"] == 0)
tag = a.tag or ("retrieval" if a.retrieval else "plain")
out = Path(f"data/bench/{tag}_n{a.n}_r{a.rounds}.json")
out.parent.mkdir(parents=True, exist_ok=True)
# {meta, trials}, not a bare list. A score with no record of the model,
# adapter and mix that produced it cannot be compared to anything -- which
# is how "one variable per experiment" stops being a property and becomes a
# slogan. Readers should accept either shape; the older files are lists.
run_meta.update({"render_success": round(n_ok / len(rows), 4),
                 "n_ok": n_ok, "first_try": first_try,
                 "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
out.write_text(json.dumps({"meta": run_meta, "trials": rows}, indent=2))

print("\n" + "="*60)
print(f"  {tag.upper()}  ({a.rounds} repair rounds, retrieval={a.retrieval})")
print("="*60)
print(f"  render success rate : {n_ok/len(rows):.1%}   ({n_ok}/{len(rows)})")
print(f"  passed first try    : {first_try}")
print(f"  rescued by repair   : {n_ok - first_try}")
print(f"  lint fires          : {sum(len(r['lint']) for r in rows)}")
print(f"  remaining failures  : {Counter(r['final_error'] for r in rows if not r['ok'])}")
print(f"  wall clock          : {(time.monotonic()-t0)/60:.1f} min")
print("="*60)
