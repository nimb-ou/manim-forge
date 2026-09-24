"""A teacher reads each synthetic arc for mathematical errors.

Synthetic arcs are one teacher's imitation of 3Blue1Brown, and teachers
slip: a planner v2 arc said subtraction's order "does not matter" and, one
sentence later, that A - B is never B - A. The planner learns whatever the
narration says. This asks a second teacher, one arc per call, whether the
narration states anything mathematically false; build_planner_windows.py
drops arcs judged ERROR. Resumable; writes data/planner/judge_done at the
end.

    ./.venv/bin/python -u scripts/judge_arcs.py
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SRC = ROOT / "data" / "planner" / "plan_synth.jsonl"
OUT = ROOT / "data" / "planner" / "arc_judgements.jsonl"

PROMPT = """Below is the narration plan for a maths explainer video, one beat
per line. Check ONLY the mathematics: is any statement false, or does the
plan contradict itself? Ignore style, pacing, and missing detail.

Answer on the first line with exactly OK or ERROR. If ERROR, add one short
line saying which beat and what is wrong.

{arc}"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default="gemini-3.7-flash")
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--done-marker", default="judge_done")
    ap.add_argument("--follow", action="store_true",
                    help="keep polling --src for new arcs until the marker "
                         "named by --wait-for exists")
    ap.add_argument("--wait-for", default="")
    a = ap.parse_args()

    from forge.synth.teacher import Teacher
    teacher = Teacher(provider=a.provider, model=a.model)

    def _impatient(signum, frame):                            # noqa: ANN001
        raise TimeoutError("teacher did not answer in time")
    signal.signal(signal.SIGALRM, _impatient)

    marker_dir = ROOT / "data" / "planner"
    bad = n = 0
    while True:
        done = ({json.loads(l)["id"] for l in a.out.open() if l.strip()}
                if a.out.exists() else set())
        rows = ([json.loads(l) for l in a.src.open() if l.strip()]
                if a.src.exists() else [])
        todo = [r for r in rows if r["meta"]["id"] not in done]
        print(f"{len(done)} judged, {len(todo)} to go", flush=True)
        if not todo:
            if a.follow and not (marker_dir / a.wait_for).exists():
                for _ in range(20):
                    time.sleep(15)
                continue
            break
        n, bad = judge(todo, teacher, a, n, bad)
    (marker_dir / a.done_marker).write_text(
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    print(f"done: {n} judged, {bad} ERROR", flush=True)
    return 0


def judge(todo, teacher, a, n, bad):
    for r in todo:
        arc = r["messages"][2]["content"].replace("\nEND", "")
        signal.alarm(120)
        try:
            reply = teacher.ask(PROMPT.format(arc=arc[:12000]), max_tokens=300,
                                system="You are a careful mathematics referee. "
                                       "Plain text only.")
        except Exception as exc:                              # noqa: BLE001
            print(f"  {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            signal.alarm(0)   # or it fires during the wait below
            # A 429 is the free daily quota; it comes back in hours, not
            # seconds, and hammering it only resets the clock.
            waits = 40 if "429" in str(exc) else 1
            for _ in range(waits):
                time.sleep(15)
            continue
        finally:
            signal.alarm(0)
        head = reply.strip().splitlines()[0].strip().upper() if reply.strip() else ""
        verdict = "ERROR" if head.startswith("ERROR") else \
            "OK" if head.startswith("OK") else "UNCLEAR"
        with a.out.open("a") as f:
            f.write(json.dumps({"id": r["meta"]["id"], "verdict": verdict,
                                "why": " ".join(reply.split())[:300],
                                "judge": f"{a.provider}:{a.model}"},
                               ensure_ascii=False) + "\n")
        n += 1
        bad += verdict == "ERROR"
        if n % 25 == 0:
            print(f"  {n} judged this run, {bad} ERROR", flush=True)
        time.sleep(a.pause)
    return n, bad


if __name__ == "__main__":
    sys.exit(main())
