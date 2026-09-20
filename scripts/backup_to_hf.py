"""Back the corpus up to a private Hugging Face dataset repo.

Chosen over cloud file storage because it is free and unlimited for datasets,
git-backed so every snapshot is versioned rather than overwritten, and built
for exactly this shape of data. Nothing new has to be installed.

**Private by default, and deliberately.** The corpus mixes sources whose terms
are stated (CC BY-NC-SA) with sources whose terms are not stated at all.
Republishing someone's data publicly under terms they never granted is not
something to do by accident, so going public is an explicit flag and should
follow a licensing review — or better, a clean public release built only from
what is unambiguously ours: the synthetic scenes, the gold scenes, and the
render verdicts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

CARD = """---
license: cc-by-nc-sa-4.0
tags: [manim, animation, code-generation, render-verified]
---

# Manim Forge corpus — {stamp}

Every row in the verified splits has been **executed and rendered**. Code that
does not produce video is not in them.

## Contents

{table}

## Why render-verified

Public Manim datasets are text that was never run. Scraped Manim code spans
years of incompatible API versions, so a model trained on it learns a
plausible-looking dialect that does not execute. Gating on an actual render is
a mechanical quality filter that needs no human judgement per row.

Of 3,680 deduplicated public rows, **1,760 (47.8%) survived** the gate.

## Provenance

Every row carries `source` and `license`. Sources with stated terms are
CC BY-NC-SA 4.0; several public datasets state no terms, and rows from those
are marked `unstated`. This snapshot is private for that reason.

Generated with a free-tier teacher model and verified locally on an M4 MacBook
Air. Project: https://github.com/nimb-ou/manim-forge
"""


def restore(api, repo: str, dest: Path) -> None:
    """Pull the backup down into ./data.

    A backup with no restore path is a hope, not a backup — and this one had
    none for the whole life of the project. It is also what makes CI possible
    at all: `data/` is gitignored, so a checkout on a runner has no corpus,
    no gate verdicts and no gold rows, and `forge.doctor` cannot check
    anything that matters without them.
    """
    from huggingface_hub import snapshot_download
    print(f"restoring {repo} -> {dest}")
    got = snapshot_download(repo_id=repo, repo_type="dataset",
                            local_dir=str(dest.parent),
                            allow_patterns=["data/**"])
    n = sum(1 for _ in Path(got).rglob("*") if _.is_file())
    print(f"  {n} files restored")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="default: <user>/manim-forge-corpus")
    ap.add_argument("--public", action="store_true",
                    help="NOT default — see the licensing note in this file")
    ap.add_argument("--include-frames", action="store_true")
    ap.add_argument("--restore", action="store_true",
                    help="download the backup into ./data instead of uploading")
    a = ap.parse_args()

    from huggingface_hub import HfApi, whoami
    from forge.catalog import CATALOG

    try:
        me = whoami()["name"]
    except Exception:
        raise SystemExit("Not authenticated. Run: ./.venv/bin/hf auth login --token <token>")

    repo = a.repo or f"{me}/manim-forge-corpus"
    api = HfApi()
    if a.restore:
        restore(api, repo, Path("data"))
        return
    api.create_repo(repo, repo_type="dataset", private=not a.public, exist_ok=True)
    print(f"repo: https://huggingface.co/datasets/{repo} "
          f"({'PUBLIC' if a.public else 'private'})")

    stamp = time.strftime("%Y-%m-%d %H:%M")
    rows = []
    uploaded = 0
    for d in CATALOG:
        p = Path(d.path)
        if not d.archive or not p.exists():
            continue
        if p.is_dir():
            if p.name == "frames" and not a.include_frames:
                rows.append(f"| `{d.path}` | (skipped — pass --include-frames) | {d.purpose} |")
                continue
            api.upload_folder(folder_path=str(p), path_in_repo=p.as_posix(),
                              repo_id=repo, repo_type="dataset")
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            rows.append(f"| `{d.path}` | {n} files | {d.purpose} |")
        else:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=p.as_posix(),
                            repo_id=repo, repo_type="dataset")
            n = sum(1 for _ in p.open())
            rows.append(f"| `{d.path}` | {n:,} rows | {d.purpose} |")
        uploaded += 1
        print(f"  uploaded {d.path}")

    table = ("| file | size | purpose |\n|---|---|---|\n" + "\n".join(rows))
    card = Path("/tmp/README.md")
    card.write_text(CARD.format(stamp=stamp, table=table))
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset")

    print(f"\n{uploaded} datasets uploaded")
    print(f"https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
