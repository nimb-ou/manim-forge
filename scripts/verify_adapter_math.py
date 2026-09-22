#!/usr/bin/env python3
"""Check that a converted MLX adapter applies the same delta as its PEFT source.

    ./.venv/bin/python scripts/verify_adapter_math.py \
        --peft adapters/kaggle-sft/adapter --mlx adapters/mlx-sft

`peft_to_mlx.py --verify` already proves the adapter *loads* -- it probes a
weight and asserts the model changed, which rules out mlx-lm's silent
`load_weights(strict=False)` no-op. It does not prove the conversion is
*correct*. The two formats disagree about orientation:

    PEFT   A (r, in), B (out, r)      delta_W = (alpha/r) . B @ A
    MLX    lora_a (in, r), lora_b (r, out)
           y = x @ W.T + scale . (x @ lora_a) @ lora_b
           so                          delta_W = scale . (lora_a @ lora_b).T

A transposed conversion produces a loaded, non-zero, wrong adapter -- one
that degrades the model while passing every check in the pipeline, and whose
damage would be read as a finding about the corpus. This compares the two
deltas numerically instead.

Run it before believing any number measured through the MLX path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

# fp16 round-trip through two matmuls; a transpose error is O(1), not O(1e-4)
TOLERANCE = 1e-3


def modules_of(peft_keys: list[str]) -> list[tuple[int, str]]:
    out = set()
    for k in peft_keys:
        if ".lora_A." not in k:
            continue
        body = k.split("model.layers.")[1]
        layer, rest = body.split(".", 1)
        out.add((int(layer), rest.rsplit(".lora_A", 1)[0]))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peft", required=True, type=Path)
    ap.add_argument("--mlx", required=True, type=Path)
    ap.add_argument("--sample", type=int, default=12,
                    help="modules to check, spread across the depth (0 = all)")
    a = ap.parse_args()

    cfg = json.loads((a.peft / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]

    P = safe_open(a.peft / "adapter_model.safetensors", framework="np")
    M = safe_open(a.mlx / "adapters.safetensors", framework="np")
    mods = modules_of(list(P.keys()))
    if a.sample and len(mods) > a.sample:
        step = len(mods) / a.sample
        mods = [mods[int(i * step)] for i in range(a.sample)]

    print(f"r={cfg['r']} alpha={cfg['lora_alpha']} scale={scale}")
    worst, worst_name, missing = 0.0, "", []
    for layer, mod in mods:
        pre = f"base_model.model.model.layers.{layer}.{mod}"
        mre = f"model.layers.{layer}.{mod}"
        try:
            A = P.get_tensor(f"{pre}.lora_A.weight").astype(np.float32)
            B = P.get_tensor(f"{pre}.lora_B.weight").astype(np.float32)
            lo_a = M.get_tensor(f"{mre}.lora_a").astype(np.float32)
            lo_b = M.get_tensor(f"{mre}.lora_b").astype(np.float32)
        except Exception as exc:                              # noqa: BLE001
            missing.append(f"{mre}: {type(exc).__name__}")
            continue
        d_peft = scale * (B @ A)
        d_mlx = scale * (lo_a @ lo_b).T
        if d_peft.shape != d_mlx.shape:
            print(f"  SHAPE  L{layer} {mod}: {d_peft.shape} vs {d_mlx.shape}")
            worst, worst_name = float("inf"), f"L{layer} {mod}"
            continue
        peak = float(np.abs(d_peft).max())
        rel = float(np.abs(d_peft - d_mlx).max()) / max(peak, 1e-12)
        flag = "ok " if rel < TOLERANCE else "BAD"
        print(f"  {flag} L{layer:<2} {mod:<20} |delta|max={peak:.5f} "
              f"relerr={rel:.1e}")
        if rel > worst:
            worst, worst_name = rel, f"L{layer} {mod}"

    if missing:
        print("\nmissing from one side:")
        for m in missing[:10]:
            print("  ", m)
        return 1
    print(f"\nworst relative error: {worst:.1e} at {worst_name} "
          f"(tolerance {TOLERANCE:g})")
    if worst >= TOLERANCE:
        print("The converted adapter does NOT apply the same delta. Any number "
              "measured through it is about the conversion, not the training.")
        return 1
    print("The converted adapter applies the same delta as its PEFT source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
