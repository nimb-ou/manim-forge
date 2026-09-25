#!/usr/bin/env python3
"""Push a training kernel, from the same script the scene run uses.

    ./.venv/bin/python scripts/push_kernel.py coder
    ./.venv/bin/python scripts/push_kernel.py planner

There is one kernel file, `kaggle/01_sft.py`, and three things to train with
it. Copying it into `kaggle/coder/` by hand would mean two files drifting
apart, and the fixes in that script cost seventeen runs to find -- a stale
copy would rediscover them.

So the copy is made at push time and the directory holds only the metadata
that differs: which dataset to mount and what to call the kernel.

Sequence length is per adapter and measured, not chosen: the coder's rows
are a median of 323 tokens so it starts at 768, and the planner's windows are
a median of 1,013 so it starts at 2048. Both substitutions are asserted --
a silent no-op would train at the scene run's length and look like it worked.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle" / "01_sft.py"

# Starting sequence length per adapter, measured rather than chosen.
#   coder    median 323 tokens, p90 441, max 1,312
#   planner  median 1,013, p90 1,143, 2.2% over 2048
#   kit      short trained prompt: see build_kit_dataset.py (measured below)
SEQ = {"coder": (768, 512), "planner": (2048, 1536), "kit": (1024, 768)}


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "coder"
    if which not in SEQ:
        print(f"unknown kernel {which!r}; expected one of {sorted(SEQ)}")
        return 2
    dest = ROOT / "kaggle" / which
    start, fallback = SEQ[which]
    text = SRC.read_text()

    # One substitution, asserted. A silent no-op here would train the coder
    # at the scene run's sequence length and look like it worked.
    old = "_attempt, _seen, _won = (True, 2048), set(), False"
    new = f"_attempt, _seen, _won = (True, {start}), set(), False"
    if text.count(old) != 1:
        print(f"expected exactly one {old!r} in {SRC}; found {text.count(old)}")
        return 1
    text = text.replace(old, new)

    old_plan = "return (use_fp16, 1536) if max_len > 1536 else None"
    if text.count(old_plan) != 1:
        print("the retry plan's descent no longer matches; not pushing")
        return 1
    text = text.replace(
        old_plan,
        f"return (use_fp16, {fallback}) if max_len > {fallback} else None")

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "01_sft.py").write_text(text)
    check = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_kernel.py"),
         str(dest / "01_sft.py")])
    if check.returncode != 0:
        return check.returncode
    return subprocess.run(
        [str(ROOT / ".venv" / "bin" / "kaggle"), "kernels", "push",
         "-p", str(dest)]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
