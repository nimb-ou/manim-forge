"""Publish a trained adapter pulled back from Kaggle.

The kernel cannot do this itself. A Kaggle notebook sees none of the
runner's environment, so HF_TOKEN would have to be attached as a Kaggle
Secret through the web UI -- a manual step in a loop whose whole point is
that it has no manual steps. The workflow already downloads the kernel's
output and already holds the token, so the push belongs there.

    python scripts/publish_adapter.py out/ --repo nimitttt/manim-forge-sft

Refuses rather than guesses: an adapter with no weights, or a directory with
no adapter in it, is an error and not an empty upload. The failure mode this
avoids is the one that keeps recurring here -- a step that reports success
while producing nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")


def find_adapter(root: Path) -> Path:
    """The directory holding adapter_config.json, wherever Kaggle put it."""
    if (root / "adapter_config.json").exists():
        return root
    found = sorted(p.parent for p in root.rglob("adapter_config.json"))
    if not found:
        raise SystemExit(
            f"no adapter_config.json under {root}\n"
            f"contents: {sorted(p.name for p in root.iterdir())[:10]}")
    if len(found) > 1:
        raise SystemExit(f"several adapters under {root}: {found}")
    return found[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="where the kernel output was pulled")
    ap.add_argument("--repo", default="nimitttt/manim-forge-sft")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    adapter = find_adapter(a.path)
    weights = [adapter / w for w in WEIGHTS if (adapter / w).exists()]
    if not weights:
        raise SystemExit(f"{adapter} has a config but no weights — "
                         f"training did not finish")
    size_mb = sum(w.stat().st_size for w in weights) / 1e6
    if size_mb < 1:
        raise SystemExit(f"adapter weights are {size_mb:.2f} MB — too small "
                         f"to be a trained LoRA; refusing to publish")

    cfg = json.loads((adapter / "adapter_config.json").read_text())
    run = {}
    if (adapter / "run.json").exists():
        run = json.loads((adapter / "run.json").read_text())

    print(f"adapter:  {adapter}")
    print(f"weights:  {size_mb:.1f} MB")
    print(f"rank:     {cfg.get('r')}  alpha: {cfg.get('lora_alpha')}")
    print(f"targets:  {cfg.get('target_modules')}")
    if run:
        print(f"mix:      {run.get('mix_sha256')}  "
              f"{run.get('train_rows')} rows, {run.get('epochs')} epochs")
    if a.dry_run:
        print("\n(dry run — nothing uploaded)")
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit("no HF_TOKEN in the environment")

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(a.repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(adapter), repo_id=a.repo)
    print(f"\npushed to https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
