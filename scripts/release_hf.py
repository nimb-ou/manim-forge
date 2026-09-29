"""Upload the app's two MLX adapters and a model card to Hugging Face.

Private by default. A public release is a separate, deliberate step
(--public), taken only once the numbers in the card are the shipped ones.

    ./.venv/bin/python scripts/release_hf.py --planner adapters/mlx-planner3 \\
        --coder adapters/mlx-coder6-kit --tag v1.0
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CARD = """---
license: cc-by-nc-sa-4.0
base_model: mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
library_name: mlx
tags: [manim, animation, mathematics, lora, mlx]
---

# Manim Forge adapters ({tag})

Two LoRA adapters on Qwen2.5-Coder-7B-Instruct (MLX, 4-bit) that turn a
plain-English request into a short 3Blue1Brown-style animation with the
[Manim Forge](https://github.com/nimb-ou/manim-forge) kit:

- `planner/` writes the arc: 3-8 beats, each a picture and its narration.
- `coder/` writes one beat's code at a time with the Forge kit's blocks.

Trained with QLoRA on Kaggle from render-verified, vision-critic-filtered
scenes. Use them through the app (`python -m forge.serve`), which embeds
the kit, renders each scene and salvages failing statements.

Held-out result (20 prompts on topics nothing was built for; share of
planned beats a vision judge says show the beat's idea): **{heldout}**.

CC BY-NC-SA 4.0: non-commercial, share-alike.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", required=True)
    ap.add_argument("--coder", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--heldout", default="see docs/RESULTS.md")
    ap.add_argument("--repo", default="manim-forge")
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi, whoami
    user = whoami()["name"]
    repo = f"{user}/{a.repo}"
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=not a.public, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        card = Path(d) / "README.md"
        card.write_text(CARD.format(tag=a.tag, heldout=a.heldout))
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                        repo_id=repo, commit_message=f"{a.tag}: model card")
    for name, src in (("planner", a.planner), ("coder", a.coder)):
        api.upload_folder(folder_path=str(ROOT / src), path_in_repo=name, repo_id=repo,
                          commit_message=f"{a.tag}: {name} from {Path(src).name}")
    print(f"uploaded to https://huggingface.co/{repo} "
          f"({'public' if a.public else 'private'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
