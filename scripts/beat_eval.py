"""The coder alone: does its beat render on top of the true earlier beats?

The eight-task split eval measures planner and coder together, eight times.
A render rate of 3/8 cannot say which half failed, and eight is too few to
tell coder v2 from v3 unless the difference is enormous.

This holds the planner out entirely. For every held-out beat in the coder's
validation split, the scene is rebuilt from the *reference* bodies of the
beats before it, the model writes this beat from the same prompt it trained
on, and the result is assembled and rendered. The reference beat is
rendered the same way first, and rows whose reference does not render are
excluded -- a beat cut out of a scene that only worked whole is not the
coder's failure.

    ./.venv/bin/python -u scripts/beat_eval.py --adapter adapters/mlx-coder2 \\
        --tag coder2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from forge.app.twostage import Beat, assemble, extract_code  # noqa: E402
from run_twostage import ask, load  # noqa: E402

VALID = ROOT / "kaggle" / "manim-forge-coder" / "valid.jsonl"


def scenes() -> dict[str, list[dict]]:
    """Valid rows by scene, keeping scenes whose beats run 0..k unbroken."""
    by = defaultdict(list)
    for l in VALID.open():
        if l.strip():
            r = json.loads(l)
            by[r["meta"]["scene"]].append(r)
    out = {}
    for s, rows in by.items():
        rows.sort(key=lambda r: int(r["meta"]["index"]))
        if [int(r["meta"]["index"]) for r in rows] == list(range(len(rows))):
            out[s] = rows
    return out


def build(bodies: list[str]):
    beats = [Beat(n=i + 1, seconds=None, intent=f"step {i + 1}")
             for i in range(len(bodies))]
    return assemble(beats, bodies)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=900)
    a = ap.parse_args()

    from forge.harness import RenderHarness
    h = RenderHarness(python_bin=str(ROOT / ".venv" / "bin" / "python"),
                      cache_dir=str(ROOT / "data" / "frames"), timeout=120)
    model, tok = load(a.adapter)

    rows_out, tally = [], Counter()
    todo = [(s, i, rows) for s, rows in sorted(scenes().items())
            for i in range(len(rows))]
    if a.limit:
        todo = todo[:a.limit]
    t0 = time.time()
    for k, (s, i, rows) in enumerate(todo, 1):
        truth = [extract_code(r["messages"][-1]["content"]) for r in rows]
        ref = build(truth[:i + 1])
        ref_ok = ref.ok and h.render(ref.code, quality="low", frames=4).ok
        if not ref_ok:
            tally["reference fails"] += 1
            continue
        row = rows[i]
        reply = ask(model, tok, row["messages"][0]["content"],
                    row["messages"][1]["content"], a.max_tokens)
        cand = extract_code(reply)
        asm = build(truth[:i] + [cand])
        res = h.render(asm.code, quality="low", frames=4) if asm.ok else None
        verdict = ("rendered" if res and res.ok else
                   res.error_kind.value if res else
                   ("unparsable" if any("parse" in p for p in asm.problems)
                    else "undefined names" if any("names" in p
                                                  for p in asm.problems)
                    else "assembly"))
        tally[verdict] += 1
        rows_out.append({"id": row["meta"]["id"], "scene": s, "index": i,
                         "verdict": verdict, "problems": asm.problems,
                         "stderr": (res.stderr[-600:] if res and not res.ok
                                    else ""),
                         "lines": len(cand.splitlines()),
                         "ref_lines": len(truth[i].splitlines())})
        scored = len(rows_out)
        print(f"  [{k}/{len(todo)}] {s}:{i} {verdict}   "
              f"running {tally['rendered']}/{scored} "
              f"({time.time() - t0:.0f}s)", flush=True)

    scored = len(rows_out)
    out = ROOT / "data" / "bench" / f"beat_eval_{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"adapter": a.adapter, "tally": tally,
                               "scored": scored, "rows": rows_out}, indent=1))
    print(f"\n{a.tag}: {tally['rendered']}/{scored} beats render "
          f"({100 * tally['rendered'] / max(scored, 1):.1f}%), "
          f"{tally['reference fails']} excluded (reference fails)")
    for k2, v in tally.most_common():
        print(f"  {k2:20s} {v}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
