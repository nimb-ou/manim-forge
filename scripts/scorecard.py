"""One scorecard for the pipeline, with the picture in it.

    ./.venv/bin/python -u scripts/scorecard.py --n 12 --kit --tag kit
    ./.venv/bin/python -u scripts/scorecard.py --n 12 --tag raw

The first long hard-eval renders scored well on coverage and length and
were slides: coverage counts terms in the code, and captions contain terms;
length counts seconds, and waits fill them. So every number here sits
beside two that a slide cannot fake:

  * visual beats -- the share of beats that build something that is not
    text: a kit block that draws (plane, vector, graph, riemann, ...) or a
    non-text Mobject;
  * a contact sheet -- six frames from each render, tiled, in
    data/scorecard/<tag>/, to be looked at before any number is believed.

Coverage is computed on the construct() body only: in kit mode the
embedded kit source is full of words like "draw_vector" and "slide_tangent".
Runs through forge.app.pipeline, the same code the demo and server use.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forge.app.pipeline import Options, SwapHost, run  # noqa: E402
from forge.app.twostage import repeats_earlier  # noqa: E402

from forge.kit.kit import KIT_BLOCKS as KIT_DRAWS  # noqa: E402
from forge.kit.kit import KIT_SCAFFOLD  # noqa: E402
TEXT = {"Text", "MathTex", "Tex", "Title", "MarkupText", "Paragraph",
        "BulletedList", "Code", "Integer", "DecimalNumber", "Variable"}


def construct_body(code: str) -> str:
    return code.split("def construct(self):", 1)[-1]


def visual(body: str, mobjects: set[str]) -> bool:
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return False
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    # A background (plane, axes) alone is not a picture: GRPO learned to
    # draw empty grids for the bonus. Content blocks or non-text Mobjects
    # beyond the scaffold count.
    return bool(calls & (KIT_DRAWS - KIT_SCAFFOLD)) \
        or bool((calls & mobjects) - TEXT - {"NumberPlane", "Axes",
                                             "NumberLine", "ComplexPlane"}) \
        or ".plot(" in body


def contact_sheet(video: str, out: Path, k: int = 6) -> None:
    ff = "/opt/homebrew/bin/ffmpeg"
    probe = subprocess.run(["/opt/homebrew/bin/ffprobe", "-v", "error",
                            "-show_entries", "format=duration", "-of",
                            "csv=p=0", video], capture_output=True, text=True)
    try:
        dur = float(probe.stdout.strip())
    except ValueError:
        return
    frames = []
    for i in range(k):
        f = out.with_suffix(f".{i}.jpg")
        subprocess.run([ff, "-loglevel", "error", "-y", "-ss",
                        f"{dur * (i + 0.5) / k:.2f}", "-i", video, "-frames:v",
                        "1", "-vf", "scale=480:-1", str(f)])
        if f.exists():
            frames.append(f)
    if len(frames) == k:
        inputs = sum([["-i", str(f)] for f in frames], [])
        half = k // 2
        graph = (f"{''.join(f'[{i}]' for i in range(half))}hstack={half}[a];"
                 f"{''.join(f'[{i}]' for i in range(half, k))}hstack={half}[b];"
                 f"[a][b]vstack")
        subprocess.run([ff, "-loglevel", "error", "-y", *inputs,
                        "-filter_complex", graph, str(out)])
    for f in frames:
        f.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--short", action="store_true",
                    help="the 20 short one-idea prompts (the headline eval) "
                         "instead of the hard titles")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--planner", default=str(ROOT / "adapters" / "mlx-planner3"))
    ap.add_argument("--coder", default=str(ROOT / "adapters" / "mlx-coder2"))
    ap.add_argument("--kit", action="store_true")
    ap.add_argument("--max-beats", type=int, default=12)
    ap.add_argument("--quality", default="low")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    import manim
    mob = {n for n in dir(manim) if isinstance(getattr(manim, n), type)
           and issubclass(getattr(manim, n), manim.Mobject)}
    from forge.evaluate.hard_eval import build_tasks, concept_coverage
    if a.short:
        from types import SimpleNamespace
        spec = json.loads((ROOT / "forge" / "evaluate" / "short_prompts.json")
                          .read_text())["prompts"]
        # No reference video: coverage is 0 and length is against a nominal
        # 60 s. The numbers that matter here are renders, visual beats and
        # the contact sheets.
        tasks = [SimpleNamespace(prompt=p["prompt"], terms=[], real_seconds=60.0)
                 for p in spec][a.start: a.start + a.n]
    else:
        tasks = build_tasks()[a.start: a.start + a.n]
    out_dir = ROOT / "data" / "scorecard" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    host = SwapHost(a.planner, a.coder)
    opts = Options(max_beats=a.max_beats, quality=a.quality, kit=a.kit)

    rows = []
    for i, t in enumerate(tasks, 1):
        t0 = time.time()
        res = run(t.prompt, host, lambda e: None, opts)
        bodies = [b for b in res.bodies if b.strip()]
        # Visual and new: a picture repeated from an earlier beat does not
        # count (GRPO drew the same plane-and-circle five times running).
        vis = sum(visual(b, mob) and not repeats_earlier(b, bodies[:k])
                  for k, b in enumerate(bodies))
        cov, hits = concept_coverage(construct_body(res.code), t.terms)
        row = {"title": t.prompt[:70], "ok": res.ok, "beats": len(res.beats),
               "kept": len(bodies), "visual_beats": vis,
               "coverage": round(cov, 3), "matched": hits,
               "seconds": res.duration or 0.0, "real_seconds": t.real_seconds,
               "error": res.error, "notes": res.notes,
               "elapsed": round(time.time() - t0)}
        slug = re.sub(r"[^a-z0-9]+", "-", t.prompt.lower())[:40].strip("-")
        (out_dir / f"{i:02d}-{slug}.py").write_text(res.code)
        if res.ok and res.video:
            contact_sheet(res.video, out_dir / f"{i:02d}-{slug}.jpg")
        rows.append(row)
        print(f"  [{i}/{len(tasks)}] ok={res.ok} beats={row['kept']}/"
              f"{row['beats']} visual={vis}/{row['kept']} cov={cov:.0%} "
              f"{res.error[:50]} ({row['elapsed']}s)", flush=True)

    ok = [r for r in rows if r["ok"]]
    kept = sum(r["kept"] for r in rows) or 1
    summary = {
        "tag": a.tag, "kit": a.kit, "planner": a.planner, "coder": a.coder,
        "n": len(rows), "rendered": len(ok),
        "visual_beat_share": round(sum(r["visual_beats"] for r in rows) / kept, 3),
        "coverage_all": round(sum(r["coverage"] for r in rows) / len(rows), 3),
        "length_ratio_rendered": round(
            sum(r["seconds"] / r["real_seconds"] for r in ok) / len(ok), 4)
        if ok else 0.0,
        "mean_beats_kept": round(kept / len(rows), 1),
    }
    (out_dir / "scorecard.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1))
    print("\n" + "  ".join(f"{k}={v}" for k, v in summary.items()
                           if k not in ("planner", "coder")))
    print(f"contact sheets: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
