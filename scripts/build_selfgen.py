"""Bundle round-2 self-training for Kaggle: arcs + the forge package, and the
kernel metadata that mounts the newest kit coder SFT output as its adapter.

Arcs: the teacher's narration and synthetic arcs (the drawable ones), the
Claude-written scenes' and plans' arcs, and Gemma's picture-naming arcs --
never a held-out topic -- up to --limit, 3-6 beats each.

    ./.venv/bin/python scripts/build_selfgen.py --limit 3000
      -> kaggle/manim-forge-selfgen/{arcs.jsonl, forge/}, kaggle/selfgen/kernel-metadata.json
    then: kaggle datasets create|version -p kaggle/manim-forge-selfgen
          kaggle kernels push -p kaggle/selfgen          (once kit v8 has trained)
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gemma_arcs import BANNED  # noqa: E402
from forge.app.twostage import parse_plan  # noqa: E402

DS = ROOT / "kaggle" / "manim-forge-selfgen"
KDIR = ROOT / "kaggle" / "selfgen"


def arcs() -> list[dict]:
    out = []
    from synth_kit_beats import arcs as teacher_arcs
    for rid, req, beats in teacher_arcs(5000):
        # the teacher's arcs keep the planner's whole user message as request
        req = req.split("REQUEST\n", 1)[-1].split("\n\n", 1)[0].strip()
        out.append(("t:" + rid, req, beats))
    for name in ("claude_plans.jsonl", "gemma_plans.jsonl"):
        f = ROOT / "data" / "kit" / name
        if f.exists():
            for l in f.read_text().splitlines():
                r = json.loads(l)
                bs, _ = parse_plan(r["messages"][2]["content"], limit=10)
                out.append((r["meta"]["id"], r["meta"]["request"], bs))
    keep = []
    for rid, req, bs in out:
        bs = bs[:6]
        if len(bs) < 3 or BANNED.search(req) or any(BANNED.search(b.intent) for b in bs):
            continue
        keep.append({"id": rid, "request": req[:300],
                     "beats": [[k + 1, b.seconds, b.intent, b.narration] for k, b in enumerate(bs)]})
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3000)
    a = ap.parse_args()
    got = arcs()
    random.Random(0).shuffle(got)
    got = got[: a.limit]
    if DS.exists():
        shutil.rmtree(DS)
    DS.mkdir(parents=True)
    (DS / "arcs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in got))
    shutil.copytree(ROOT / "forge", DS / "forge",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "teacher", "gold"))
    (DS / "dataset-metadata.json").write_text(json.dumps(
        {"title": "manim-forge-selfgen", "id": "nimbou/manim-forge-selfgen",
         "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}))
    (KDIR / "kernel-metadata.json").write_text(json.dumps({
        "id": "nimbou/manim-forge-selfgen", "title": "Manim Forge selfgen",
        "code_file": "selfgen.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": ["nimbou/manim-forge-selfgen"], "competition_sources": [],
        "kernel_sources": ["nimbou/manim-forge-kit-sft"]}, indent=1))
    print(f"{len(got)} arcs -> {DS / 'arcs.jsonl'}; forge/ copied; kernel metadata written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
