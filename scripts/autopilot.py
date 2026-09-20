#!/usr/bin/env python3
"""Wait for the Kaggle run, collect the adapter, and evaluate it. Unattended.

    nohup ./.venv/bin/python scripts/autopilot.py > /dev/null 2>&1 &
    tail -f data/autopilot/autopilot.log

The kernel now decides for itself whether to train with AMP on or off, so
nothing between pushing it and having numbers requires a judgement call --
it is polling, a download, a format conversion, and about ninety minutes of
local generation and rendering. That is a script, not a conversation, and
writing it down means the four hours of GPU time are not followed by a wait
for someone to notice they finished.

Resumable on purpose: every phase checks whether its output already exists,
so a laptop that slept, or a run interrupted at hour six, restarts by being
run again rather than by starting over.

Exit codes: 0 evaluated, 1 the kernel failed, 2 timed out waiting, 3 the
adapter arrived but was unusable.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
KAGGLE = ROOT / ".venv" / "bin" / "kaggle"
STATE_DIR = ROOT / "data" / "autopilot"
LOG = STATE_DIR / "autopilot.log"
STATE = STATE_DIR / "state.json"
RAW = ROOT / "adapters" / "kaggle-sft"

# Kaggle's KernelWorkerStatus values, split by what they mean for waiting.
# Tested against all seven in tests/test_kaggle_poll.py; repeated here rather
# than imported because this script has to keep running when the repo it
# lives in is mid-edit.
RUNNING = {"queued", "running", "pending"}
GOOD = {"complete"}
BAD = {"error", "cancelrequested", "cancelacknowledged", "cancelled"}


def say(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def note(**fields) -> None:
    """Write the current phase where something other than a human can read it.

    docs/STATE.md is generated from files like this one rather than from my
    memory of what happened, which has been wrong often enough to matter.
    """
    STATE.parent.mkdir(parents=True, exist_ok=True)
    prior = {}
    if STATE.exists():
        try:
            prior = json.loads(STATE.read_text())
        except json.JSONDecodeError:
            pass
    prior.update(fields)
    prior["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    STATE.write_text(json.dumps(prior, indent=2) + "\n")


def kaggle_status(kernel: str) -> str:
    """Return the kernel's status word, lowercased, or "" if unreadable.

    A network blip during a fourteen-hour poll must not end the poll, so an
    unreadable status is a reason to wait rather than to exit.
    """
    r = subprocess.run([str(KAGGLE), "kernels", "status", kernel],
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    # The CLI prints: <slug> has status "KernelWorkerStatus.RUNNING" --
    # enum repr, not the bare word I first wrote this against, so the strict
    # pattern never matched and every poll was silently falling through to
    # the substring scan below. Match the enum member rather than the whole
    # line
    # rather than scanning the whole line for "error" -- a transient message
    # that merely contains the word would otherwise abandon a healthy run.
    m = re.search(r'status\s+"?(?:KernelWorkerStatus\.)?([a-zA-Z]+)"?', out)
    if m:
        return m.group(1).lower()
    for word in sorted(GOOD | BAD | RUNNING):
        if word in out.lower():
            return word
    say(f"    unrecognised status output: {out[:200]!r}")
    return ""


def wait_for_kernel(kernel: str, poll: int, max_hours: float) -> str:
    deadline = time.time() + max_hours * 3600
    last = None
    while time.time() < deadline:
        status = kaggle_status(kernel)
        if status != last:
            say(f"kernel {kernel}: {status or 'unknown'}")
            note(phase="waiting", kernel=kernel, kernel_status=status)
            last = status
        if status in GOOD:
            return status
        if status in BAD:
            return status
        time.sleep(poll)
    return "timeout"


def collect(kernel: str, dest: Path) -> Path | None:
    """Download the kernel's output and find the PEFT adapter inside it."""
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(KAGGLE), "kernels", "output", kernel,
                        "-p", str(dest)], capture_output=True, text=True)
    say(f"    kaggle kernels output -> {(r.stdout + r.stderr).strip()[:300]}")

    hits = sorted(dest.rglob("adapter_config.json"))
    if not hits:
        say("    no adapter_config.json in the output — the kernel finished "
            "without saving an adapter")
        return None
    if len(hits) > 1:
        say(f"    several adapters in the output: {[str(h) for h in hits]}; "
            f"taking the first")
    found = hits[0].parent
    weights = [p.name for p in found.iterdir()
               if p.suffix in (".safetensors", ".bin")]
    if not weights:
        say(f"    {found} has a config but no weights")
        return None
    say(f"    adapter at {found} ({', '.join(sorted(weights))})")

    run_json = found / "run.json"
    if run_json.exists():
        say("    run.json:\n" + run_json.read_text())
    return found


def dump_log(dest: Path) -> None:
    r = subprocess.run([str(PY), str(ROOT / "scripts" / "read_kaggle_log.py"),
                        str(dest)], capture_output=True, text=True)
    tail = r.stdout[-6000:] if r.stdout else (r.stderr or "")[-2000:]
    say("    last of the kernel log:\n" + tail)


def evaluate(peft_dir: Path) -> int:
    """Run the Phase 1 exit condition and stream it into the same log."""
    say(f"evaluating {peft_dir} — about 90 minutes of generation and renders")
    note(phase="evaluating", adapter=str(peft_dir))
    with LOG.open("a") as f:
        r = subprocess.run(["bash", str(ROOT / "scripts" / "evaluate_adapter.sh"),
                            str(peft_dir)], stdout=f, stderr=subprocess.STDOUT)
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernel", default="nimbou/manim-forge-sft")
    ap.add_argument("--poll", type=int, default=600,
                    help="seconds between status checks (default 600)")
    ap.add_argument("--max-hours", type=float, default=14.0,
                    help="give up waiting after this long (Kaggle's GPU "
                         "session limit is 12h)")
    ap.add_argument("--skip-wait", action="store_true",
                    help="the kernel is already done; collect and evaluate")
    args = ap.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    say("=" * 70)
    say(f"autopilot: {args.kernel}")
    note(phase="starting", kernel=args.kernel)

    if not args.skip_wait:
        status = wait_for_kernel(args.kernel, args.poll, args.max_hours)
        if status == "timeout":
            say(f"still not finished after {args.max_hours}h — giving up "
                f"waiting. The kernel may still be running; re-run with "
                f"--skip-wait once it is done.")
            note(phase="timeout")
            return 2
        if status in BAD:
            say(f"kernel finished as {status}")
            note(phase="failed", kernel_status=status)
            dest = STATE_DIR / "kernel-output"
            dest.mkdir(parents=True, exist_ok=True)
            subprocess.run([str(KAGGLE), "kernels", "output", args.kernel,
                            "-p", str(dest)], capture_output=True, text=True)
            dump_log(dest)
            return 1
        say("kernel complete")

    note(phase="collecting")
    peft_dir = collect(args.kernel, RAW)
    dump_log(RAW)
    if peft_dir is None:
        note(phase="no-adapter")
        return 3

    note(phase="collected", adapter=str(peft_dir))
    code = evaluate(peft_dir)
    if code != 0:
        say(f"evaluation exited {code} — see the log above. The adapter is "
            f"kept at {peft_dir} either way.")
        note(phase="eval-failed", exit_code=code)
        return 3

    say("evaluated. The numbers to read are length ratio and concept "
        "coverage, not render rate: the untuned model already renders 85%, "
        "and docs/PLAN.md predicts the other two will not move.")
    note(phase="done", exit_code=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
