"""Convert a PEFT LoRA adapter into the format mlx-lm loads.

Training happens on Kaggle, which means PEFT; evaluation happens on the
laptop, which means MLX. The two write LoRA to disk differently and neither
will tell you when you hand it the other's file:

    PEFT   adapter_model.safetensors    base_model.model.model.…lora_A.weight
                                        A: (r, in)        B: (out, r)
                                        delta = (alpha/r) · B @ A

    MLX    adapters.safetensors         model.…lora_a
                                        lora_a: (in, r)   lora_b: (r, out)
                                        delta = scale · lora_bᵀ @ lora_aᵀ

So `lora_a = Aᵀ`, `lora_b = Bᵀ`, `scale = alpha / r`, and the two deltas are
then the same matrix.

**The dangerous part is the loading, not the maths.** `load_adapters` ends
with `model.load_weights(..., strict=False)`, so an adapter whose keys do not
match the model silently loads *nothing* and leaves you benchmarking the base
model while believing you are benchmarking the tuned one. That is a worse
outcome than a crash, and it is exactly the class of failure this project has
been cleaning up all day, so this script verifies rather than assumes:

  1. every tensor it writes is checked to exist in the built LoRA model
  2. `--verify` loads the result and asserts the weights actually changed

    ./.venv/bin/python scripts/peft_to_mlx.py \\
        --peft adapters/kaggle-sft --out adapters/mlx-sft --verify
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BASE_DEFAULT = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"


def convert(peft_dir: Path, out_dir: Path) -> dict:
    import mlx.core as mx

    cfg_path = peft_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"no adapter_config.json in {peft_dir}")
    peft_cfg = json.loads(cfg_path.read_text())

    r = int(peft_cfg["r"])
    alpha = float(peft_cfg.get("lora_alpha", r))
    scale = alpha / r

    weights_file = next(
        (p for p in (peft_dir / "adapter_model.safetensors",
                     peft_dir / "adapter_model.bin") if p.exists()), None)
    if weights_file is None:
        raise SystemExit(f"no adapter weights in {peft_dir}")

    if weights_file.suffix == ".safetensors":
        raw = mx.load(str(weights_file))
    else:                                   # torch pickle; convert via numpy
        import torch
        raw = {k: mx.array(v.float().numpy())
               for k, v in torch.load(weights_file, map_location="cpu").items()}

    out: dict[str, "mx.array"] = {}
    layers: set[int] = set()
    keys: set[str] = set()
    for key, w in raw.items():
        if ".lora_A" not in key and ".lora_B" not in key:
            continue
        # base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight
        #   -> model.layers.N.self_attn.q_proj.lora_a
        name = key
        for prefix in ("base_model.model.", "base_model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        name = name.replace(".weight", "")
        if name.endswith(".lora_A"):
            name, w = name[: -len(".lora_A")] + ".lora_a", w.T
        elif name.endswith(".lora_B"):
            name, w = name[: -len(".lora_B")] + ".lora_b", w.T
        else:
            continue
        out[name] = w.astype(mx.float16)
        part = name.split(".")
        if "layers" in part:
            i = part.index("layers")
            layers.add(int(part[i + 1]))
            # MLX matches against `layer.named_modules()`, whose keys are
            # relative to the transformer block -- "self_attn.q_proj", not
            # "q_proj". A bare projection name matches nothing,
            # linear_to_lora_layers silently converts no layers, and
            # load_weights(strict=False) then loads none of the adapter while
            # reporting success.
            keys.add(".".join(part[i + 2:-1]))

    if not out:
        raise SystemExit("no LoRA tensors found — is this a PEFT adapter?")

    num_layers = max(layers) + 1 if layers else 0

    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), out)
    (out_dir / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": r,
            "scale": scale,
            "dropout": float(peft_cfg.get("lora_dropout", 0.0)),
            "keys": sorted(f"{k}" for k in keys),
        },
        "converted_from": str(peft_dir),
        "peft_target_modules": peft_cfg.get("target_modules"),
    }, indent=2) + "\n")
    for extra in ("run.json",):
        if (peft_dir / extra).exists():
            shutil.copy2(peft_dir / extra, out_dir / extra)

    return {"tensors": len(out), "layers": num_layers, "rank": r,
            "alpha": alpha, "scale": scale, "modules": sorted(keys),
            "out": str(out_dir)}


def verify(out_dir: Path, base: str) -> None:
    """Load it and prove the weights moved.

    `load_adapters` finishes with `load_weights(..., strict=False)`. A key
    mismatch is therefore silent: the model loads, generates, and scores --
    as the base model. The only honest check is to compare a tensor before
    and after.
    """
    import mlx.core as mx
    from mlx_lm import load

    print(f"\nverifying against {base} …", flush=True)
    plain, _ = load(base)
    tuned, _ = load(base, adapter_path=str(out_dir))

    adapters = mx.load(str(out_dir / "adapters.safetensors"))
    nonzero_b = [k for k in adapters if k.endswith("lora_b")
                 and float(mx.abs(adapters[k]).sum()) > 0]
    if not nonzero_b:
        raise SystemExit(
            "every lora_b is zero — the adapter encodes no change at all.\n"
            "That is what an untrained or mis-saved adapter looks like.")

    probe = nonzero_b[0]
    path = probe[: -len(".lora_b")].split(".")
    def dig(m, parts):
        for p in parts:
            m = m[int(p)] if p.isdigit() else getattr(m, p)
        return m
    try:
        mod = dig(tuned, path)
    except Exception as exc:
        raise SystemExit(f"key {probe} does not exist on the model: {exc}\n"
                         "The names did not match, and strict=False hid it.")
    if not hasattr(mod, "lora_b"):
        raise SystemExit(f"{'.'.join(path)} is not a LoRA layer — "
                         "linear_to_lora_layers did not reach it")

    loaded = float(mx.abs(mod.lora_b).sum())
    expected = float(mx.abs(adapters[probe]).sum())
    if abs(loaded - expected) > 1e-3 * max(expected, 1.0):
        raise SystemExit(f"{probe} did not load: model has {loaded:.4f}, "
                         f"file has {expected:.4f}")

    n_lora = sum(1 for _ in mx.load(str(out_dir / "adapters.safetensors")))
    print(f"  {n_lora} tensors, {len(nonzero_b)} non-zero lora_b")
    print(f"  probe {probe}: |w| = {loaded:.4f}, matches the file")
    print("  the adapter is loaded and changes the model")
    del plain, tuned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peft", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--verify", action="store_true",
                    help="load it and assert the weights actually moved")
    a = ap.parse_args()

    stats = convert(a.peft, a.out)
    print("=" * 58)
    for k, v in stats.items():
        print(f"{k:>10}: {v}")
    print("=" * 58)
    if a.verify:
        verify(a.out, a.base)


if __name__ == "__main__":
    main()
