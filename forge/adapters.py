"""Guard against loading an adapter mlx-lm will silently ignore.

`mlx_lm.load(..., adapter_path=X)` ends in
`model.load_weights(..., strict=False)`. Hand it something whose keys do not
match and it loads *nothing*, generates fine, and scores as the base model.
There is no error and no warning -- the benchmark simply reports the
untuned number under the tuned name.

Two ways to arrive there, both of which happened here:

* a PEFT adapter straight from Kaggle. Different filename, different key
  names, different tensor orientation. See scripts/peft_to_mlx.py.
* a converted adapter whose `keys` are bare projection names. MLX matches
  those against `layer.named_modules()`, whose keys are relative to the
  transformer block -- "self_attn.q_proj", not "q_proj" -- so nothing
  matches and no layer becomes a LoRA layer.

So this refuses before the model loads, rather than after the numbers are
in.
"""

from __future__ import annotations

import json
from pathlib import Path


def check_mlx_adapter(path: str | Path | None) -> None:
    """Raise SystemExit unless `path` is an adapter mlx-lm will really load."""
    if not path:
        return
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"adapter not found: {p}")

    if (p / "adapter_model.safetensors").exists() and not (
            p / "adapters.safetensors").exists():
        raise SystemExit(
            f"{p} is a PEFT adapter; mlx-lm cannot read it, and it loads with\n"
            f"strict=False so it would silently score the base model.\n"
            f"Convert it first:\n"
            f"  ./.venv/bin/python scripts/peft_to_mlx.py \\\n"
            f"      --peft {p} --out {p}-mlx --verify")

    weights = p / "adapters.safetensors"
    cfg_file = p / "adapter_config.json"
    if not weights.exists() or not cfg_file.exists():
        raise SystemExit(
            f"{p} is not an MLX adapter — expected adapters.safetensors and "
            f"adapter_config.json, found {sorted(x.name for x in p.iterdir())}")

    cfg = json.loads(cfg_file.read_text())
    keys = (cfg.get("lora_parameters") or {}).get("keys")
    if keys is not None:
        bare = [k for k in keys if "." not in k]
        if bare:
            raise SystemExit(
                f"{p} has bare projection names in lora_parameters.keys: "
                f"{bare}\nMLX matches these against layer.named_modules(), "
                f"which yields 'self_attn.q_proj' -- bare names match nothing "
                f"and the adapter loads as a no-op.\nRe-convert with "
                f"scripts/peft_to_mlx.py.")
    if not cfg.get("num_layers"):
        raise SystemExit(f"{p}: adapter_config.json has no num_layers; "
                         f"linear_to_lora_layers would convert nothing")
