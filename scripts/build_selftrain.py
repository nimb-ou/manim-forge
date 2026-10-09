"""Self-training rows (data/selfgen/rows.jsonl) -> the Kaggle dataset.

A row is kept when its scene rendered whole with nothing the critic could
find, and was not rejected by eye or by the scene judge (``--reject`` file:
one row number per line, as graded from data/selfgen/sheets/). The system
prompt is the app's (kit reference + lesson plan); the user prompt is the
one the scene was written from.

    ./.venv/bin/python scripts/build_selftrain.py [--reject data/eye/selfgen_reject.txt]
      -> kaggle/manim-forge-selftrain/{train,valid}.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reject", default="")
    ap.add_argument("--eye", default="",
                    help="data/eye/selfgen.json: keep only rows graded G there")
    a = ap.parse_args()
    from forge.app.oneshot import system_prompt
    from forge.evaluate.heldout_guard import touches_heldout
    rows = [json.loads(x) for x in (ROOT / "data" / "selfgen" / "rows.jsonl")
            .read_text().splitlines() if x.strip()]
    reject = set()
    if a.reject:
        reject = {int(x.split()[0]) for x in Path(a.reject).read_text().split("\n")
                  if x.strip() and not x.startswith("#")}
    system = system_prompt(True, plan=True)
    # Problems the critic no longer reports (a grid past the frame edge, from
    # rows written before the kit stopped counting it) do not disqualify.
    stale = re.compile(r"a (NumberPlane|ComplexPlane) runs off the edge")
    for r in rows:
        r["clean"] = bool(r.get("ok")) and not [p for p in r.get("problems", [])
                                               if not stale.search(p)]
    if a.eye:
        eye = json.loads(Path(a.eye).read_text())
        reject |= {r["i"] for r in rows if eye.get(str(r["i"]), ["?"])[0] != "G"}
    keep = [r for r in rows if r.get("clean") and r["i"] not in reject
            and r.get("user") and not touches_heldout(r["request"])]
    out = [{"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": r["user"]},
                         {"role": "assistant", "content": r["reply"].strip()}],
            "meta": {"request": r["request"], "i": r["i"]}} for r in keep]
    random.Random(5).shuffle(out)
    n_valid = max(4, len(out) // 20)
    dest = ROOT / "kaggle" / "manim-forge-selftrain"
    dest.mkdir(parents=True, exist_ok=True)
    for name, part in (("valid", out[:n_valid]), ("train", out[n_valid:])):
        (dest / f"{name}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in part))
    print(f"{len(rows)} tried, {sum(r.get('clean', False) for r in rows)} clean, "
          f"{len(reject)} rejected by eye -> {len(out) - n_valid} train / {n_valid} valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
