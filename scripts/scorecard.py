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


def contact_sheet(video: str, out: Path, k: int = 6, times=None) -> None:
    """Frames in a grid of three columns: at each beat's end when ``times``
    (the pipeline's beat marks) are known, else ``k`` evenly spaced."""
    import shutil
    ff = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    probe = subprocess.run([shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe", "-v", "error",
                            "-show_entries", "format=duration", "-of",
                            "csv=p=0", video], capture_output=True, text=True)
    try:
        dur = float(probe.stdout.strip())
    except ValueError:
        return
    at = [max(t - 0.15, 0) for t in times] if times else \
        [dur * (i + 0.5) / k for i in range(k)]
    frames = []
    for i, t in enumerate(at):
        f = out.with_suffix(f".{i}.jpg")
        subprocess.run([ff, "-loglevel", "error", "-y", "-ss", f"{min(t, dur - 0.05):.2f}",
                        "-i", video, "-frames:v", "1", "-vf", "scale=480:-1", str(f)])
        if f.exists():
            frames.append(f)
    if frames:
        while len(frames) % 3:                       # pad the last row
            frames.append(frames[-1])
        inputs = sum([["-i", str(f)] for f in frames], [])
        rows = [f"{''.join(f'[{i}]' for i in range(r, r + 3))}hstack=3[r{r}]"
                for r in range(0, len(frames), 3)]
        n = len(rows)
        graph = ";".join(rows) + (";" + "".join(f"[r{r}]" for r in range(0, len(frames), 3))
                                  + f"vstack={n}" if n > 1 else "")
        if n == 1:
            graph = graph.replace("[r0]", "")
        subprocess.run([ff, "-loglevel", "error", "-y", *inputs,
                        "-filter_complex", graph, str(out)])
    for f in set(frames):
        f.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heldout", action="store_true",
                    help="the 20 held-out prompts (forge/evaluate/heldout_prompts.json)")
    ap.add_argument("--fresh", action="store_true",
                    help="the 20 fresh test requests (forge/evaluate/fresh_prompts.json)")
    ap.add_argument("--world", action="store_true",
                    help="the 30 real-world requests (forge/evaluate/world_prompts.json)")
    ap.add_argument("--inscope", action="store_true",
                    help="the 20 in-scope prompts (forge/evaluate/inscope_prompts.json)")
    ap.add_argument("--oneshot", action="store_true",
                    help="write the whole scene in one call from the nearest "
                         "hand-written scenes (forge/app/oneshot.py); --coder "
                         "is the adapter ('none' for the base model)")
    ap.add_argument("--api", action="store_true",
                    help="one-shot: put the kit reference in the system prompt")
    ap.add_argument("--k", type=int, default=2, help="one-shot: examples shown")
    ap.add_argument("--base", default="", help="one-shot: base model (default "
                    "forge.app.oneshot.BASE_MODEL)")
    ap.add_argument("--plan", action="store_true",
                    help="one-shot: write a lesson plan (# plan: lines) before the beats")
    ap.add_argument("--think", action="store_true",
                    help="one-shot: let the model reason (enable_thinking) first")
    ap.add_argument("--revise", type=int, default=0,
                    help="one-shot: attempts after the first that rewrite the best "
                         "draft with its problems listed (forge/app/critique.py)")
    ap.add_argument("--check", action="store_true",
                    help="one-shot: the model checks the numbers on screen (critique.self_check)")
    ap.add_argument("--ids", default="",
                    help="only these prompts, by number (1-based, comma-separated)")
    ap.add_argument("--samples", type=int, default=1,
                    help="one-shot: render up to N samples, keep the best")
    ap.add_argument("--short", action="store_true",
                    help="the 20 short one-idea prompts (the headline eval) "
                         "instead of the hard titles")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--planner", default=str(ROOT / "adapters" / "mlx-planner3"))
    ap.add_argument("--coder", default=str(ROOT / "adapters" / "mlx-coder2"))
    ap.add_argument("--kit", action="store_true")
    ap.add_argument("--plans", default="",
                    help="a plan cache (JSON): prompts found there use the cached "
                         "beats, others are planned once and added -- so every "
                         "coder configuration is scored on identical plans")
    ap.add_argument("--signatures", action="store_true",
                    help="relevance hint gives each block's call (off: names only)")
    ap.add_argument("--exemplar", action="store_true",
                    help="show the coder the nearest hand-written scene")
    ap.add_argument("--relevance", action="store_true",
                    help="subject hint in the prompt + resample off-subject beats")
    ap.add_argument("--max-beats", type=int, default=12)
    ap.add_argument("--quality", default="low")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    import manim
    mob = {n for n in dir(manim) if isinstance(getattr(manim, n), type)
           and issubclass(getattr(manim, n), manim.Mobject)}
    from forge.evaluate.hard_eval import build_tasks, concept_coverage
    if a.short or a.heldout or a.inscope or a.world or a.fresh:
        from types import SimpleNamespace
        name = "heldout_prompts.json" if a.heldout else \
            "inscope_prompts.json" if a.inscope else \
            "world_prompts.json" if a.world else \
            "fresh_prompts.json" if a.fresh else "short_prompts.json"
        spec = json.loads((ROOT / "forge" / "evaluate" / name)
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
    if a.oneshot:
        from forge.app.oneshot import run_oneshot
        from forge.app.pipeline import load
        from forge.app.oneshot import BASE_MODEL
        om, otok = load(None if a.coder in ("", "none") else a.coder,
                        base=a.base or BASE_MODEL)
        a.kit = True
    else:
        host = SwapHost(a.planner, a.coder)
    opts = Options(max_beats=a.max_beats, quality=a.quality, kit=a.kit, mark_beats=True,
                   relevance=a.relevance, exemplar=a.exemplar,
                   signatures=a.signatures)

    rows = []
    ids = {int(x) for x in a.ids.split(",") if x.strip()}
    for i, t in enumerate(tasks, a.start + 1):
        if ids and i not in ids:
            continue
        t0 = time.time()
        # The planner samples (temperature 0.5): unseeded, two runs of the
        # same prompt got different plans and per-prompt swings of 0/6 to
        # 4/6 swamped the comparison. Seeded per prompt, every
        # configuration with the same planner gets the same plan.
        import mlx.core as mx
        mx.random.seed(1000 + i)
        given = None
        if a.plans:
            from forge.app.twostage import Beat
            cache_path = Path(a.plans)
            cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
            if t.prompt not in cache:
                from forge.app.pipeline import plan as _plan
                pm, ptok = host.use("planner")
                got = _plan(pm, ptok, t.prompt, opts.stride, opts.max_beats)
                cache[t.prompt] = [[b.n, b.seconds, b.intent, b.narration] for b in got]
                cache_path.write_text(json.dumps(cache, indent=1))
            given = [Beat(*x) for x in cache[t.prompt]]
        if a.oneshot:
            res = run_oneshot(t.prompt, om, otok, opts=opts, api=a.api, k=a.k,
                              samples=a.samples, plan=a.plan, think=a.think,
                              revise=a.revise, check=a.check)
        else:
            res = run(t.prompt, host, lambda e: None, opts, beats=given)
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
               "layout": getattr(res, "layout", []),
               "issues": getattr(res, "issues", []),
               "elapsed": round(time.time() - t0)}
        slug = re.sub(r"[^a-z0-9]+", "-", t.prompt.lower())[:40].strip("-")
        (out_dir / f"{i:02d}-{slug}.py").write_text(res.code)
        if res.ok and res.video:
            contact_sheet(res.video, out_dir / f"{i:02d}-{slug}.jpg",
                          times=res.beat_ends)
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
    # The honest number: visual, new and about its beat, over planned beats
    # (rescore.py) -- GRPO v2's 76% visual was a plane and a vector per beat.
    from rescore import score as _rescore
    import manim
    mob = {n for n in dir(manim) if isinstance(getattr(manim, n), type)
           and issubclass(getattr(manim, n), manim.Mobject)}
    rs = _rescore(a.tag, mob)
    summary["relevant_share"] = rs["share_relevant"]
    (out_dir / "scorecard.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(f"relevant_share={rs['share_relevant']}  (visual, new and about the "
          f"beat, over {rs['planned']} planned beats)")
    print(f"contact sheets: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
