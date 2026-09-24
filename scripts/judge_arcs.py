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
    a = ap.parse_args()

    from forge.synth.teacher import Teacher
    teacher = Teacher(provider=a.provider, model=a.model)

    def _impatient(signum, frame):                            # noqa: ANN001
        raise TimeoutError("teacher did not answer in time")
    signal.signal(signal.SIGALRM, _impatient)

    done = ({json.loads(l)["id"] for l in OUT.open() if l.strip()}
            if OUT.exists() else set())
    rows = [json.loads(l) for l in SRC.open() if l.strip()]
    todo = [r for r in rows if r["meta"]["id"] not in done]
    print(f"{len(done)} judged, {len(todo)} to go", flush=True)
    bad = n = 0
    for r in todo:
        arc = r["messages"][2]["content"].replace("\nEND", "")
        signal.alarm(120)
        try:
            reply = teacher.ask(PROMPT.format(arc=arc[:12000]), max_tokens=300,
                                system="You are a careful mathematics referee. "
                                       "Plain text only.")
        except Exception as exc:                              # noqa: BLE001
            print(f"  {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            time.sleep(a.pause * 10)
            continue
        finally:
            signal.alarm(0)
        head = reply.strip().splitlines()[0].strip().upper() if reply.strip() else ""
        verdict = "ERROR" if head.startswith("ERROR") else \
            "OK" if head.startswith("OK") else "UNCLEAR"
        with OUT.open("a") as f:
            f.write(json.dumps({"id": r["meta"]["id"], "verdict": verdict,
                                "why": " ".join(reply.split())[:300],
                                "judge": f"{a.provider}:{a.model}"},
                               ensure_ascii=False) + "\n")
        n += 1
        bad += verdict == "ERROR"
        if n % 25 == 0:
            print(f"  {n} judged this run, {bad} ERROR", flush=True)
        time.sleep(a.pause)
    (ROOT / "data" / "planner" / "judge_done").write_text(
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    print(f"done: {n} judged, {bad} ERROR", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
