"""Render gallery candidates exactly as the app does, timed.

    ./.venv/bin/python -u scripts/make_gallery.py
      -> docs/gallery/NN-slug.mp4, NN-slug.jpg (contact sheet), runs.json

Same path as forge.serve: the release library, Qwen3.5-9B untuned with the
kit reference, best of 2, narrated. Every candidate is kept on disk; which
ten make the gallery is decided by eye (docs/GALLERY.md).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.environ.setdefault("FORGE_LIBRARY", "release")

PROMPTS = [
    "a jacket costs £80 and is 15% off. what do you pay?",
    "if 9 people all shake hands with each other, how many handshakes is that?",
    "why do the angles of a hexagon add up to 720 degrees?",
    "how do you add 3/4 and 1/6?",
    "find the area under y = 3x from 0 to 4",
    "what does multiplying by i do to the point 2 + i?",
    "what's the chance of two heads when you flip two coins?",
    "solve 4x + 3 = 19 using a balance",
    "why is the area of a circle pi r squared?",
    "what does a 2x2 matrix do to the plane?",
    "the derivative as the slope of the tangent line",
    "how far does a bike wheel with a 35 cm radius roll in one turn?",
    "fitting a straight line to data with least squares",
    "why does a stretched spring pull back harder the further you pull it?",
]


def main() -> int:
    from forge.app.oneshot import BASE_MODEL, run_oneshot
    from forge.app.pipeline import Options, load
    from scorecard import contact_sheet
    out = ROOT / "docs" / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else {}
    t0 = time.time()
    model, tok = load(None, base=BASE_MODEL)
    print(f"model loaded in {time.time() - t0:.0f} s", flush=True)
    # --retry 06 09 ...: those scenes again from sampled (not greedy) answers,
    # best of 3 by score(), for candidates that were wrong by eye.
    retry = sys.argv[sys.argv.index("--retry") + 1:] if "--retry" in sys.argv else []
    for i, prompt in enumerate(PROMPTS, 1):
        slug = f"{i:02d}-" + re.sub(r"[^a-z0-9]+", "-", prompt.lower())[:40].strip("-")
        if retry and f"{i:02d}" not in retry:
            continue
        if not retry and slug in runs and runs[slug].get("ok"):
            continue
        t1 = time.time()
        import mlx.core as mx
        mx.random.seed(7 + i)
        res = run_oneshot(prompt, model, tok, opts=Options(
            kit=True, narrate=True, mark_beats=True, quality="medium"),
            api=True, samples=3 if retry else 2, greedy_first=not retry)
        row = {"prompt": prompt, "ok": res.ok, "seconds": round(time.time() - t1),
               "beats": [b.intent for b in res.beats], "error": res.error,
               "notes": res.notes, "video_s": res.duration}
        if res.ok and res.video:
            shutil.copy(res.video, out / f"{slug}.mp4")
            contact_sheet(res.video, out / f"{slug}.jpg", times=res.beat_ends)
            (out / f"{slug}.py").write_text(res.code.split("def construct(self):", 1)[-1])
        if retry:
            row["retry"] = True
        runs[slug] = row
        runs_path.write_text(json.dumps(runs, indent=1, ensure_ascii=False))
        print(f"[{i}/{len(PROMPTS)}] ok={res.ok} {row['seconds']} s "
              f"({len(res.beats)} beats) {prompt[:50]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
