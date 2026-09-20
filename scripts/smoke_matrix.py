"""Push several variants of the training kernel at once and see which survives.

Serial debugging against a remote GPU costs one hypothesis per cycle, and
this project has spent thirteen cycles that way. Kaggle will run kernels
under different slugs concurrently, so competing fixes can be tested in the
same six minutes instead of six times six.

Each variant is the real kernel with one edit applied. They are smoke runs --
20 steps on 64 rows -- so the whole matrix costs a few minutes of the weekly
GPU quota and answers which fix is right rather than whether one guess was.

    python scripts/smoke_matrix.py --push
    python scripts/smoke_matrix.py --status
    python scripts/smoke_matrix.py --logs
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle" / "01_sft.py"
WORK = Path("/tmp/smoke_matrix")

#: name -> (description, [(find, replace), ...])
#: Every edit is asserted to apply, so a variant can never silently be the
#: control -- which would look like a passing hypothesis.
VARIANTS = {
    "clipfix": (
        "override _clip_grad_norm and normalise gradients there (primary)",
        []),
    "noclip": (
        "max_grad_norm=0 — skip clipping entirely; does scaler.step() still unscale?",
        [("    max_grad_norm=1.0,", "    max_grad_norm=0.0,")]),
    "noamp": (
        "fp16=False — no GradScaler at all; slower, but the error cannot occur",
        [("    fp16=True, bf16=False,", "    fp16=False, bf16=False,")]),
}


def build(name: str) -> Path:
    desc, edits = VARIANTS[name]
    src = KERNEL.read_text()
    src = src.replace("SMOKE = False  # workflow-managed",
                      "SMOKE = True  # workflow-managed")
    assert "SMOKE = True" in src, "smoke flag did not apply"
    for find, repl in edits:
        assert find in src, f"{name}: anchor not found: {find!r}"
        src = src.replace(find, repl, 1)
    d = WORK / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "01_sft.py").write_text(src)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"nimbou/mf-smoke-{name}",
        "title": f"MF smoke {name}",
        "code_file": "01_sft.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": ["nimbou/manim-forge-data"],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2) + "\n")
    import ast
    ast.parse((d / "01_sft.py").read_text())      # never push what will not parse
    return d


def status(name: str) -> str:
    r = subprocess.run(["kaggle", "kernels", "status", f"nimbou/mf-smoke-{name}"],
                       capture_output=True, text=True)
    m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", r.stdout + r.stderr)
    return m.group(1) if m else "UNKNOWN"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--logs", action="store_true")
    ap.add_argument("--only", nargs="*", default=list(VARIANTS))
    a = ap.parse_args()
    names = [n for n in a.only if n in VARIANTS]

    if a.push:
        if WORK.exists():
            shutil.rmtree(WORK)
        for n in names:
            d = build(n)
            r = subprocess.run(["kaggle", "kernels", "push", "-p", str(d)],
                               capture_output=True, text=True)
            ok = "successfully pushed" in (r.stdout + r.stderr)
            print(f"  {'pushed ' if ok else 'FAILED '} {n:<9} {VARIANTS[n][0]}")
            if not ok:
                print("      " + (r.stdout + r.stderr).strip()[-200:])
        return

    if a.status:
        for n in names:
            print(f"  {n:<9} {status(n)}")
        return

    if a.logs:
        for n in names:
            out = WORK / f"out-{n}"
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True, exist_ok=True)
            subprocess.run(["kaggle", "kernels", "output",
                            f"nimbou/mf-smoke-{n}", "-p", str(out)],
                           capture_output=True, text=True)
            print(f"\n{'=' * 66}\n  {n}: {VARIANTS[n][0]}\n{'=' * 66}")
            subprocess.run([sys.executable,
                            str(ROOT / "scripts" / "read_kaggle_log.py"), str(out)])
        return

    ap.print_help()


if __name__ == "__main__":
    main()
