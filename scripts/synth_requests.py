"""Short versions of the planner's long requests.

The planner's synthetic arcs are keyed on corpus requests, which are whole
paragraphs ("I would like to create an educational math animation that
explains ... with voiceover ..."). The hard eval's prompts have a median of
six words, and so will a person typing into the app. A planner that has only
ever seen paragraphs is being tested on titles.

For each synthetic arc this asks a teacher for two rewrites of its request:
a terse one a person would type (3-10 words) and one plain sentence. The arc
is unchanged -- it answers all three -- and build_planner_windows.py spreads
arcs across the three lengths.

Ten requests per call. Resumable: ids already written are skipped.

    ./.venv/bin/python -u scripts/synth_requests.py --provider gemini \\
        --model gemini-3.7-flash
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
OUT = ROOT / "data" / "planner" / "request_variants.jsonl"
BATCH = 10

PROMPT = """Below are {k} requests people wrote for a maths animation. For each,
write two shorter versions that ask for the same animation:

  short: what someone would type into a search box, 3 to 10 words, no
         "create an animation" preamble (e.g. "why the harmonic series
         diverges", "eigenvectors as directions that stay put")
  sentence: one plain sentence, at most 25 words, the way a curious student
         would ask a teacher

Drop anything about voiceover, colours, classrooms or video length; keep the
mathematics. Answer ONLY with lines in exactly this form, one pair per request:

  1 short: ...
  1 sentence: ...
  2 short: ...

REQUESTS
{items}
"""

LINE = re.compile(r"^\s*(\d+)\s*[.):]?\s*(short|sentence)\s*:\s*(.+?)\s*$",
                  re.I)


def parse(reply: str, k: int) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for line in reply.splitlines():
        m = LINE.match(line.replace("**", ""))
        if not m:
            continue
        i = int(m.group(1))
        if 1 <= i <= k:
            out.setdefault(i, {})[m.group(2).lower()] = \
                m.group(3).strip().strip('"')
    return {i: v for i, v in out.items()
            if "short" in v and "sentence" in v
            and 2 <= len(v["short"].split()) <= 14
            and len(v["sentence"].split()) <= 40}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default="gemini-3.7-flash")
    ap.add_argument("--pause", type=float, default=4.0)
    ap.add_argument("--follow", action="store_true",
                    help="keep polling plan_synth.jsonl for new arcs")
    a = ap.parse_args()

    from forge.synth.teacher import Teacher
    teacher = Teacher(provider=a.provider, model=a.model)

    def _impatient(signum, frame):                            # noqa: ANN001
        raise TimeoutError("teacher did not answer in time")
    signal.signal(signal.SIGALRM, _impatient)

    while True:
        done = ({json.loads(l)["id"] for l in OUT.open() if l.strip()}
                if OUT.exists() else set())
        todo = []
        for l in SRC.open():
            if not l.strip():
                continue
            meta = json.loads(l)["meta"]
            if meta["id"] not in done and meta.get("request"):
                todo.append((meta["id"], meta["request"]))
        print(f"{len(done)} done, {len(todo)} to go", flush=True)
        if not todo:
            if not a.follow:
                return 0
            for _ in range(40):
                time.sleep(15)
            continue
        made = failed = 0
        for s in range(0, len(todo), BATCH):
            chunk = todo[s:s + BATCH]
            items = "\n".join(f"{i}. {' '.join(r.split())[:900]}"
                              for i, (_, r) in enumerate(chunk, 1))
            signal.alarm(180)
            try:
                # An explicit system prompt: with none, Teacher falls back to
                # its Manim-scene one and half the replies came back wrapped
                # in a ForgeScene, truncated before the last pair.
                reply = teacher.ask(PROMPT.format(k=len(chunk), items=items),
                                    max_tokens=2500, system=(
                                        "You rewrite requests concisely. "
                                        "Plain text lines only, no code."))
            except Exception as exc:                          # noqa: BLE001
                failed += 1
                print(f"  {type(exc).__name__}: {str(exc)[:160]}", flush=True)
                time.sleep(a.pause * 5)
                continue
            finally:
                signal.alarm(0)
            got = parse(reply, len(chunk))
            with OUT.open("a") as f:
                for i, v in got.items():
                    f.write(json.dumps({"id": chunk[i - 1][0], **v,
                                        "teacher": f"{a.provider}:{a.model}"},
                                       ensure_ascii=False) + "\n")
            made += len(got)
            print(f"  +{len(got)}/{len(chunk)} (total new {made}, "
                  f"failed calls {failed})", flush=True)
            time.sleep(a.pause)
        if not a.follow:
            return 0


if __name__ == "__main__":
    sys.exit(main())
