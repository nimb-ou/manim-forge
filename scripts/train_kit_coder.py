"""Train the kit coder on this Mac with MLX, starting from coder v2.

The Kaggle GPU quota is spent until the weekly reset, and the kit changes
what the coder should learn: calls to forge/kit, not raw Manim. mlx-lm can
LoRA-train the 4-bit 7B on an M4 with gradient checkpointing, so this does
not wait.

  * data: data/kit/kit_beats_clean.jsonl -- teacher kit beats (synth_kit_beats.py)
    from scenes that rendered, filtered for relevance and novelty
    (filter_kit_beats.py) -- split by scene;
  * starts from adapters/mlx-coder2 (--resume-adapter-file) with the same
    LoRA shape (rank 16, scale 2.0, the seven projections, all 28 layers),
    so SwapHost can still hot-swap it against the planner;
  * loss on the completion only (mask_prompt): the long kit system prompt
    is context, not a target;
  * writes adapters/mlx-coder5-kit.

    ./.venv/bin/python -u scripts/train_kit_coder.py --min-rows 1500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "kit" / "kit_beats_clean.jsonl"
DATA = ROOT / "data" / "kit" / "train_v5"
BASE_ADAPTER = ROOT / "adapters" / "mlx-coder2"
OUT = ROOT / "adapters" / "mlx-coder5-kit"
MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"


def rows() -> list[dict]:
    """The filtered rows, re-filtered from the teacher's output each call:
    relevance and novelty (filter_kit_beats.py) drop about half."""
    subprocess.run([sys.executable, str(ROOT / "scripts" / "filter_kit_beats.py")],
                   capture_output=True)
    return [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()] \
        if SRC.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-rows", type=int, default=1500)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-seq", type=int, default=3072)
    ap.add_argument("--wait", action="store_true",
                    help="poll until --min-rows exist instead of exiting")
    a = ap.parse_args()

    while len(data := rows()) < a.min_rows:
        if not a.wait:
            print(f"{len(data)} rows, need {a.min_rows}")
            return 1
        print(f"  {len(data)}/{a.min_rows} clean kit rows", flush=True)
        for _ in range(40):
            time.sleep(15)

    valid_scenes = {r["meta"]["scene"] for r in data
                    if int(hashlib.sha256(r["meta"]["scene"].encode())
                           .hexdigest(), 16) % 20 == 0}
    DATA.mkdir(parents=True, exist_ok=True)
    tr = [r for r in data if r["meta"]["scene"] not in valid_scenes]
    va = [r for r in data if r["meta"]["scene"] in valid_scenes] or tr[:20]
    for name, part in (("train", tr), ("valid", va)):
        (DATA / f"{name}.jsonl").write_text("".join(
            json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n"
            for r in part))
    iters = int(len(tr) * a.epochs)
    print(f"{len(tr)} train / {len(va)} valid rows; {iters} iterations", flush=True)

    base = json.loads((BASE_ADAPTER / "adapter_config.json").read_text())
    lp = base["lora_parameters"]
    cfg = {
        "model": MODEL, "train": True, "data": str(DATA),
        "fine_tune_type": "lora", "num_layers": base["num_layers"],
        "lora_parameters": {"rank": lp["rank"], "scale": lp["scale"],
                            "dropout": lp.get("dropout", 0.05),
                            "keys": lp["keys"]},
        "batch_size": 1, "iters": iters, "learning_rate": a.lr,
        "max_seq_length": a.max_seq, "grad_checkpoint": True,
        "mask_prompt": True, "steps_per_report": 20, "steps_per_eval": 200,
        "val_batches": 25, "save_every": 200,
        "resume_adapter_file": str(BASE_ADAPTER / "adapters.safetensors"),
        "adapter_path": str(OUT),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path = DATA / "lora.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    r = subprocess.run([sys.executable, "-m", "mlx_lm.lora", "-c", str(cfg_path)])
    if r.returncode:
        return r.returncode
    # The same shape as coder v2, recorded the way the other MLX adapters are,
    # so SwapHost's shape check passes.
    cfg_out = json.loads((OUT / "adapter_config.json").read_text())
    cfg_out["trained_from"] = str(BASE_ADAPTER)
    cfg_out["data"] = str(SRC)
    (OUT / "adapter_config.json").write_text(json.dumps(cfg_out, indent=2))
    shutil.copy2(cfg_path, OUT / "lora.yaml")
    print(f"-> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
