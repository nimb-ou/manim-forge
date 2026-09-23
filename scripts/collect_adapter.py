#!/usr/bin/env python3
"""Wait for a Kaggle run, collect its adapter, convert it, and prove it.

    ./.venv/bin/python scripts/collect_adapter.py \
        --kernel nimbou/manim-forge-coder-sft \
        --peft adapters/kaggle-coder --mlx adapters/mlx-coder

`autopilot.py` does this for the scene run and only for that one. The split
has two more adapters and neither had anything waiting for it, so each would
have sat finished on Kaggle until a person noticed -- which is the gap this
whole day has been about closing.

Three checks, each of which has caught a real failure here:

  * the weights read back. A download broke mid-stream once and left
    adapter_model.safetensors at zero bytes, which passed a "the file
    exists" check and died an hour later in the converter.
  * the conversion applies the same delta. PEFT and MLX disagree about
    orientation, and a transposed adapter loads, is non-zero, passes the
    probe, and quietly degrades the model.
  * a probe weight moves the model, which rules out mlx-lm's
    load_weights(strict=False) silently loading nothing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
KAGGLE = ROOT / ".venv" / "bin" / "kaggle"


def say(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%SZ}] {msg}", flush=True)


def status(kernel: str) -> str:
    r = subprocess.run([str(KAGGLE), "kernels", "status", kernel],
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr).lower()
    for word in ("complete", "error", "cancel", "running", "queued"):
        if word in out:
            return word
    return "unknown"


def weights_ok(found: Path) -> tuple[bool, str]:
    """Reuses the autopilot's check rather than growing a second one."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mf_autopilot", ROOT / "scripts" / "autopilot.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mf_autopilot"] = mod
    spec.loader.exec_module(mod)
    return mod.weights_are_readable(found)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--peft", required=True, type=Path)
    ap.add_argument("--mlx", required=True, type=Path)
    ap.add_argument("--poll", type=int, default=600)
    ap.add_argument("--max-hours", type=float, default=14.0)
    ap.add_argument("--tries", type=int, default=3)
    a = ap.parse_args()

    deadline = time.time() + a.max_hours * 3600
    last = None
    while time.time() < deadline:
        st = status(a.kernel)
        if st != last:
            say(f"{a.kernel}: {st}")
            last = st
        if st == "complete":
            break
        if st in ("error", "cancel"):
            say(f"finished as {st}; nothing to collect")
            return 1
        time.sleep(a.poll)
    else:
        say(f"still not finished after {a.max_hours}h")
        return 2

    for attempt in range(1, a.tries + 1):
        a.peft.mkdir(parents=True, exist_ok=True)
        # Bounded. The first attempt at coder v2 sat for six hours having
        # downloaded only README.md: alive, zero CPU, blocked in the Kaggle
        # client with no timeout of its own. A download that has not
        # finished in fifteen minutes is not going to, and the retry loop
        # below is a better answer than waiting.
        try:
            subprocess.run([str(KAGGLE), "kernels", "output", a.kernel,
                            "-p", str(a.peft)], capture_output=True,
                           text=True, timeout=900)
        except subprocess.TimeoutExpired:
            say(f"  attempt {attempt}: download timed out after 15 minutes")
            continue
        hits = sorted(a.peft.rglob("adapter_config.json"))
        if not hits:
            say("no adapter in the output")
            return 3
        found = hits[0].parent
        good, detail = weights_ok(found)
        say(f"  attempt {attempt}: {detail}")
        if good:
            break
        for q in found.iterdir():
            if q.stem == "adapter_model":
                q.unlink()
        if attempt == a.tries:
            return 3
        time.sleep(10)

    say("converting to MLX")
    if subprocess.run([str(PY), str(ROOT / "scripts" / "peft_to_mlx.py"),
                       "--peft", str(found), "--out", str(a.mlx),
                       "--verify"]).returncode != 0:
        return 4
    say("checking the delta survives the conversion")
    if subprocess.run([str(PY), str(ROOT / "scripts" / "verify_adapter_math.py"),
                       "--peft", str(found), "--mlx", str(a.mlx),
                       "--sample", "6"]).returncode != 0:
        return 5

    run = found / "run.json"
    if run.exists():
        say("run.json:\n" + run.read_text())
    say(f"collected and verified -> {a.mlx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
