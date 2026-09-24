"""Correct the synthetic arcs a referee flagged, rather than discard them.

judge_arcs.py flags about seven arcs in ten -- and the sampled reasons are
mostly real: a polar integral without its Jacobian, d'Alembert's formula
wrong, an undamped oscillator's phase portrait drawn as a spiral. Dropping
them would leave a few hundred arcs. This gives the teacher its own arc and
the referee's note and asks for the same plan with the mathematics
corrected; judge_arcs.py then re-checks the result, and only fixes the
second judge passes reach training.

Follows arc_judgements.jsonl as it grows; stops after judge_done exists and
every ERROR has been attempted.

    ./.venv/bin/python -u scripts/fix_arcs.py --provider mistral \\
        --model mistral-medium-latest
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from synth_plans import LINE, SYSTEM, retime  # noqa: E402

D = ROOT / "data" / "planner"
SRC, JUDGED = D / "plan_synth.jsonl", D / "arc_judgements.jsonl"
OUT = D / "plan_synth_fixed.jsonl"

PROMPT = """A mathematics referee reviewed this explainer plan and wrote:

  {why}

Rewrite the plan so every statement is mathematically correct. Keep the
same numbered-beat format, the same order and roughly the same number of
beats; change only what must change, and fix anything else you notice is
false. End with END on its own line.

PLAN
{arc}"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="mistral")
    ap.add_argument("--model", default="mistral-medium-latest")
    ap.add_argument("--pause", type=float, default=2.0)
    a = ap.parse_args()

    from forge.synth.teacher import Teacher
    teacher = Teacher(provider=a.provider, model=a.model)

    def _impatient(signum, frame):                            # noqa: ANN001
        raise TimeoutError("teacher did not answer in time")
    signal.signal(signal.SIGALRM, _impatient)

    arcs = {json.loads(l)["meta"]["id"]: json.loads(l)
            for l in SRC.open() if l.strip()}
    tried: set[str] = set()
    if OUT.exists():
        tried = {json.loads(l)["meta"]["id"] for l in OUT.open() if l.strip()}
    made = failed = 0
    while True:
        flagged = [json.loads(l) for l in JUDGED.open() if l.strip()] \
            if JUDGED.exists() else []
        todo = [j for j in flagged if j["verdict"] == "ERROR"
                and j["id"] not in tried and j["id"] in arcs]
        if not todo:
            if (D / "judge_done").exists():
                break
            for _ in range(20):
                time.sleep(15)
            continue
        for j in todo:
            tried.add(j["id"])
            r = arcs[j["id"]]
            old = [l for l in r["messages"][2]["content"].splitlines()
                   if LINE.match(l)]
            signal.alarm(240)
            try:
                reply = teacher.ask(PROMPT.format(
                    why=j["why"].removeprefix("ERROR").strip(),
                    arc="\n".join(old)), max_tokens=max(2500, len(old) * 220),
                    system=SYSTEM)
            except Exception as exc:                          # noqa: BLE001
                failed += 1
                print(f"  {type(exc).__name__}: {str(exc)[:160]}", flush=True)
                time.sleep(a.pause * 5)
                continue
            finally:
                signal.alarm(0)
            beats = [retime(l.rstrip()) for l in reply.splitlines()
                     if LINE.match(l)]
            if len(beats) < max(6, int(0.7 * len(old))):
                failed += 1
                continue
            with OUT.open("a") as f:
                f.write(json.dumps({
                    "messages": [r["messages"][0], r["messages"][1],
                                 {"role": "assistant",
                                  "content": "\n".join(beats) + "\nEND"}],
                    "meta": {**r["meta"], "n_beats": len(beats),
                             "fixed_by": f"{a.provider}:{a.model}"},
                }, ensure_ascii=False) + "\n")
            made += 1
            if made % 20 == 0:
                print(f"  {made} fixed, {failed} failed", flush=True)
            time.sleep(a.pause)
    (D / "fix_done").write_text(
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    print(f"done: {made} fixed, {failed} failed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
