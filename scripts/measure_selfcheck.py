"""Does the model's self-check (forge/app/critique.self_check) find what the
eye found? Run on self-training rows graded by eye (data/eye/selfgen.json):
a check worth having flags the rows with wrong numbers and leaves the good
ones alone.

    ./.venv/bin/python -u scripts/measure_selfcheck.py   -> data/selfgen/selfcheck.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from forge.app.critique import self_check
    from forge.app.oneshot import BASE_MODEL, parse_scene
    from forge.app.pipeline import load
    eye = json.loads((ROOT / "data" / "eye" / "selfgen.json").read_text())
    rows = {json.loads(x)["i"]: json.loads(x) for x in
            (ROOT / "data" / "selfgen" / "rows.jsonl").read_text().splitlines()}
    model, tok = load(None, base=BASE_MODEL)
    out = {}
    for k, (grade, why) in eye.items():
        if not k.isdigit() or int(k) not in rows:
            continue
        r = rows[int(k)]
        beats, bodies = parse_scene(r["reply"])
        got = self_check(r["request"], beats, bodies, model, tok)
        out[k] = {"grade": grade, "eye": why, "flags": got}
        print(f"{k:>4} {grade} flags={len(got)}  {why[:60]}", flush=True)
        for f in got:
            print("        ", f[:150], flush=True)
    (ROOT / "data" / "selfgen" / "selfcheck.json").write_text(json.dumps(out, indent=1))
    for g in "GPB":
        xs = [v for v in out.values() if v["grade"] == g]
        if xs:
            print(f"{g}: {sum(bool(v['flags']) for v in xs)} of {len(xs)} flagged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
