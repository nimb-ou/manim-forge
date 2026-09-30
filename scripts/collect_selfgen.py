"""Collect a round-2 self-training run (kaggle/selfgen): rows into
data/kit/self_beats.jsonl, each kept scene's beat-end frames judged by the
local vision critic into data/kit/critic_local.jsonl -- so the rows count
only where the critic has looked (filter_kit_beats.py).

    ./.venv/bin/python scripts/collect_selfgen.py            # download + judge
    ./.venv/bin/python scripts/collect_selfgen.py --dir D    # an output already on disk
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")
SELF = ROOT / "data" / "kit" / "self_beats.jsonl"
CRITIC = ROOT / "data" / "kit" / "critic_local.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="")
    ap.add_argument("--threshold", type=float, default=0.8)
    a = ap.parse_args()
    d = Path(a.dir) if a.dir else ROOT / "data" / "kit" / "selfgen_out"
    if not a.dir:
        d.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([KAGGLE, "kernels", "output", "nimbou/manim-forge-selfgen",
                            "-p", str(d)], capture_output=True, text=True)
        print((r.stdout + r.stderr)[-300:])
    scenes = next(iter(sorted(d.rglob("scenes.jsonl"))), None)
    if scenes is None:
        print("no scenes.jsonl in the output")
        return 1
    frames = scenes.parent / "frames"
    have = {json.loads(l)["meta"]["id"] for l in SELF.open() if l.strip()} if SELF.exists() else set()
    judged = {json.loads(l)["scene"] for l in CRITIC.open() if l.strip()} if CRITIC.exists() else set()
    from forge.evaluate.local_vision import LocalJudge
    j = None
    import re
    n_rows = n_judged = 0
    with SELF.open("a") as out, CRITIC.open("a") as crit:
        for l in scenes.open():
            s = json.loads(l)
            rows = [r for r in s["rows"] if r["meta"]["id"] not in have]
            for r in rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_rows += len(rows)
            if not s["rows"] or not s.get("live"):
                continue
            scene = s["rows"][0]["meta"]["scene"]
            if scene in judged:
                continue
            stem = re.sub(r"[^A-Za-z0-9_-]", "_", s["arc"])[:80]
            paths = [frames / f"{stem}_{k}.jpg" for k in s["live"]]
            if not all(p.exists() for p in paths):
                continue
            j = j or LocalJudge()
            ps = [j.p_yes(s["request"], s["intents"][k], p) for k, p in zip(s["live"], paths)]
            crit.write(json.dumps({"scene": scene,
                                   "verdicts": {k: "YES" if p >= a.threshold else "NO"
                                                for k, p in zip(s["live"], ps)},
                                   "p": {k: round(p, 4) for k, p in zip(s["live"], ps)},
                                   "threshold": a.threshold, "rendered": "kaggle-selfgen"}) + "\n")
            n_judged += 1
    print(f"{n_rows} rows -> {SELF.name}; {n_judged} scenes judged -> {CRITIC.name}")
    if not a.dir:
        shutil.rmtree(frames, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
