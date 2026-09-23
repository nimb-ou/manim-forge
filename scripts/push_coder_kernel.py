#!/usr/bin/env python3
"""Push the coder training kernel, from the same script the scene run uses.

    ./.venv/bin/python scripts/push_coder_kernel.py

There is one kernel file, `kaggle/01_sft.py`, and two things to train with
it. Copying it into `kaggle/coder/` by hand would mean two files drifting
apart, and the fixes in that script cost seventeen runs to find -- a stale
copy would rediscover them.

So the copy is made at push time and the directory holds only the metadata
that differs: which dataset to mount and what to call the kernel.

The coder's rows are short -- median 323 tokens against the scene corpus's
1,071 -- so `max_len` is lowered here. At 2048 nothing truncates either way;
the point is memory and speed, and 768 keeps the whole p99 while leaving the
card far emptier than the scene run, which spent five runs fighting for it.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle" / "01_sft.py"
DEST = ROOT / "kaggle" / "coder"


def main() -> int:
    text = SRC.read_text()

    # One substitution, asserted. A silent no-op here would train the coder
    # at the scene run's sequence length and look like it worked.
    old = "_attempt, _seen, _won = (True, 2048), set(), False"
    new = "_attempt, _seen, _won = (True, 768), set(), False"
    if text.count(old) != 1:
        print(f"expected exactly one {old!r} in {SRC}; found {text.count(old)}")
        return 1
    text = text.replace(old, new)

    old_plan = "return (use_fp16, 1536) if max_len > 1536 else None"
    if text.count(old_plan) != 1:
        print("the retry plan's descent no longer matches; not pushing")
        return 1
    text = text.replace(old_plan, "return (use_fp16, 512) if max_len > 512 else None")

    (DEST / "01_sft.py").write_text(text)
    check = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_kernel.py"),
         str(DEST / "01_sft.py")])
    if check.returncode != 0:
        return check.returncode
    return subprocess.run(
        [str(ROOT / ".venv" / "bin" / "kaggle"), "kernels", "push",
         "-p", str(DEST)]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
