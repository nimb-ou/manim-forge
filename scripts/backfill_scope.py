#!/usr/bin/env python3
"""Add NAMES IN SCOPE to coder rows that were built without it.

    ./.venv/bin/python scripts/backfill_scope.py

The untuned two-stage run failed with

    beats use names no beat defines: Diagram, radius, show_smaller_angle

-- the coder inventing objects because nothing told it what already existed.
Listing intents is not enough: "a circle appears" does not say the circle is
called `c`.

The fix belongs in the prompt, and `decompose_corpus.py` now writes it. But
re-running that means 1,296 teacher calls to regenerate labels that are
already correct, and the scope itself is computed locally from code the file
already contains. So this rewrites the rows in place instead: group by
scene, walk the beats in order, and derive each one's scope from the answers
before it.

Free, and it keeps the labels that were paid for.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from forge.app.twostage import extract_code, names_in_scope

ROOT = Path(__file__).resolve().parents[1]
MARK = "\nNAMES IN SCOPE\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", type=Path,
                    default=ROOT / "data" / "planner" / "coder_corpus.jsonl")
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.path.read_text().splitlines() if l.strip()]
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scene[r["meta"]["scene"]].append(r)

    done = skipped = 0
    for group in by_scene.values():
        group.sort(key=lambda r: r["meta"]["index"])
        bodies: list[str] = []
        for r in group:
            user = r["messages"][1]["content"]
            if MARK in user:
                skipped += 1
            else:
                scope = names_in_scope(bodies)
                inject = (MARK + "  "
                          + (", ".join(scope) if scope else "(none yet)")
                          + "\n")
                # Before HELPERS, so the order matches what the builders emit
                # and the two sources stay one format.
                anchor = "\nHELPERS THIS SCENE DEFINES\n"
                r["messages"][1]["content"] = (
                    user.replace(anchor, inject + anchor, 1)
                    if anchor in user else user + inject)
                done += 1
            bodies.append(extract_code(r["messages"][2]["content"]))

    a.path.write_text("\n".join(json.dumps(r) for g in by_scene.values()
                                for r in g) + "\n")
    print(f"{done} rows given a scope, {skipped} already had one -> {a.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
