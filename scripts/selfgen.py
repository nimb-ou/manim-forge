"""Training rows from the model's own best scenes (v1.5 phase 5).

Every fine-tune so far trained the model on someone else's scenes and
cost it reasoning (docs/HISTORY.md). These rows are its own: the shipped
engine (plan, best of 2, one critique-and-rewrite) writes a scene for a
library request with that request's own scene withheld, and a row is kept
only when the render is whole and the critic finds nothing
(forge/app/critique.py). What a fine-tune on them can learn is to do first
time what the rewrite does second time.

Requests come from the evaluation library (held-out topics already out),
minus any close to a dev, test, in-scope or held-out prompt, so no
evaluation set sees its own topic in training.

    FORGE_LIBRARY=eval ./.venv/bin/python -u scripts/selfgen.py --n 400
      -> data/selfgen/rows.jsonl (one line per request tried), sheets/NNN.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.environ["FORGE_LIBRARY"] = "eval"
OUT = ROOT / "data" / "selfgen"
EVAL_SETS = ("world_prompts", "fresh_prompts", "inscope_prompts", "heldout_prompts")


def eval_prompts() -> list[str]:
    out = []
    for name in EVAL_SETS:
        spec = json.loads((ROOT / "forge" / "evaluate" / f"{name}.json").read_text())
        out += [p["prompt"] if isinstance(p, dict) else p for p in spec["prompts"]]
    return out


def requests(limit: float = 0.25) -> list[str]:
    """Library requests not close to any evaluation prompt (TF-IDF cosine
    over the library's own word split)."""
    import math
    from collections import Counter
    from forge.evaluate.heldout_guard import touches_heldout
    from forge.kit.library import _words, scenes
    reqs = [s["request"] for s in scenes() if not touches_heldout(s["request"])]
    held = eval_prompts()
    docs = [Counter(_words(t)) for t in reqs + held]
    df = Counter(w for d in docs for w in d)
    n = len(docs)

    def vec(d):
        v = {w: c * math.log(n / df[w]) for w, c in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}
    hv = [vec(d) for d in docs[len(reqs):]]
    keep = []
    for r, d in zip(reqs, docs):
        v = vec(d)
        if max(sum(x * h.get(w, 0.0) for w, x in v.items()) for h in hv) < limit:
            keep.append(r)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=15)
    a = ap.parse_args()
    from forge.app.critique import problems
    from forge.app.oneshot import BASE_MODEL, run_oneshot
    from forge.app.pipeline import Options, load
    from scorecard import contact_sheet
    pool = requests()
    random.Random(a.seed).shuffle(pool)
    pool = pool[: a.n]
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    path = OUT / "rows.jsonl"
    done = {json.loads(x)["request"] for x in path.read_text().splitlines()} \
        if path.exists() else set()
    print(f"{len(pool)} requests ({len(done)} done)", flush=True)
    model, tok = load(None, base=BASE_MODEL)
    import mlx.core as mx
    for i, req in enumerate(pool, 1):
        if req in done:
            continue
        mx.random.seed(a.seed + i)
        t0 = time.time()
        res = run_oneshot(req, model, tok, opts=Options(
            kit=True, narrate=False, mark_beats=True, quality="low"),
            api=True, samples=2, revise=1, plan=True, exclude={req})
        found = problems(res) if res.beats else ["no beats"]
        from forge.app.oneshot import user_prompt
        row = {"i": i, "request": req, "ok": res.ok, "reply": res.reply,
               "user": user_prompt(req, 2, exclude={req}),
               "beats": [b.intent for b in res.beats], "problems": found,
               "clean": res.ok and not found,
               "revised": any("(revised)" in n for n in res.notes),
               "notes": res.notes, "seconds": round(time.time() - t0)}
        if res.ok and res.video:
            contact_sheet(res.video, OUT / "sheets" / f"{i:03d}.jpg", times=res.beat_ends)
        with path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(pool)}] clean={row['clean']} revised={row['revised']} "
              f"{row['seconds']} s  {req[:60]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
