"""New explainer topics, phrased the way a person would ask.

The corpus's 1,514 requests are nearly used up by synth_plans.py, and most of
them describe a drawing ("a square moves right") rather than an idea to
explain. This asks a teacher for mathematical topics across areas and levels,
each written as a request between one line and a short paragraph, so the
planner sees what the app will actually be asked. Deduplicated by lowercase
text; resumable.

    ./.venv/bin/python -u scripts/synth_topics.py --target 1500
"""
from __future__ import annotations

import argparse
import json
import random
import re
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "planner" / "topics.jsonl"

AREAS = [
    "arithmetic and number sense", "fractions and ratios", "algebra",
    "functions and graphs", "geometry", "trigonometry", "precalculus",
    "limits and continuity", "differential calculus", "integral calculus",
    "sequences and series", "multivariable calculus", "vector calculus",
    "linear algebra", "differential equations", "complex numbers",
    "probability", "statistics", "combinatorics", "number theory",
    "graph theory", "topology", "group theory", "abstract algebra",
    "real analysis", "Fourier analysis", "information theory",
    "algorithms and computation", "cryptography", "neural networks",
    "physics-flavoured mathematics (waves, orbits, heat)", "geometry of higher dimensions",
    "mathematical puzzles and paradoxes", "famous proofs",
]
STYLES = [
    "a short search-style phrase of 3-8 words",
    "one plain question a curious student would ask",
    "one or two sentences asking for an intuitive visual explanation",
    "a short paragraph (3-4 sentences) describing what the viewer should come away understanding",
]
PROMPT = """List {k} distinct topics in {area} that would make a good
3Blue1Brown-style animated explanation, at a mix of levels from high school
to early graduate. Write each as a request someone would type to an
animation app, phrased as {style}. Ask for an idea to be explained, not a
drawing to be made. No numbering commentary, no preamble: one request per
line, each line starting with "- "."""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default="gemini-3.7-flash")
    ap.add_argument("--target", type=int, default=1500)
    ap.add_argument("--pause", type=float, default=4.0)
    a = ap.parse_args()

    from forge.synth.teacher import Teacher
    teacher = Teacher(provider=a.provider, model=a.model)

    def _impatient(signum, frame):                            # noqa: ANN001
        raise TimeoutError("teacher did not answer in time")
    signal.signal(signal.SIGALRM, _impatient)

    seen = set()
    if OUT.exists():
        seen = {json.loads(l)["request"].lower() for l in OUT.open() if l.strip()}
    rng = random.Random(len(seen))
    while len(seen) < a.target:
        area, style = rng.choice(AREAS), rng.choice(STYLES)
        signal.alarm(180)
        try:
            reply = teacher.ask(PROMPT.format(k=15, area=area, style=style),
                                max_tokens=2500, system=(
                                    "You suggest maths topics. Plain text "
                                    "lines only, no code."))
        except Exception as exc:                              # noqa: BLE001
            print(f"  {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            time.sleep(a.pause * 5)
            continue
        finally:
            signal.alarm(0)
        new = 0
        with OUT.open("a") as f:
            for line in reply.splitlines():
                m = re.match(r"^\s*[-*•]\s+(.+?)\s*$", line.replace("**", ""))
                if not m:
                    continue
                req = m.group(1).strip().strip('"')
                if len(req.split()) < 3 or req.lower() in seen:
                    continue
                seen.add(req.lower())
                f.write(json.dumps({"request": req, "area": area,
                                    "style": style,
                                    "teacher": f"{a.provider}:{a.model}"},
                                   ensure_ascii=False) + "\n")
                new += 1
        print(f"  {area} / {style[:24]}: +{new} (total {len(seen)})",
              flush=True)
        time.sleep(a.pause)
    return 0


if __name__ == "__main__":
    sys.exit(main())
