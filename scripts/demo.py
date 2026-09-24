"""Type a request, watch both models work, get a video. Local only, no API.

    ./.venv/bin/python scripts/demo.py "why the harmonic series diverges"
    ./.venv/bin/python scripts/demo.py            # asks for the request

Stage 1, the planner adapter, writes the arc beat by beat. Stage 2, the
coder adapter, writes each beat's Manim code with the names earlier beats
built. The beats are assembled into one scene, checked, and rendered; the
video opens when it is done. Everything runs on this Mac through MLX.

Options: --beats N (default 8), --quality low|medium|high (default medium),
--planner / --coder to pick adapters, --no-open.
"""
from __future__ import annotations

import argparse
import ast
import gc
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from forge.app.twostage import (assemble, extract_code,  # noqa: E402
                                beat_prompt, failing_beat,
                                missing_names, prelude_prompt)
from run_twostage import CODE_SYSTEM, ask, load, plan  # noqa: E402

B, D, G, R, Y, C, X = ("\033[1m", "\033[2m", "\033[32m", "\033[31m",
                       "\033[33m", "\033[36m", "\033[0m")


def free() -> None:
    """Return a dropped model's memory. The caller must `del` its names
    first: deleting a function's own parameters frees nothing, and the first
    version held both 7B models at once -- 9 GB and into swap."""
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:                                         # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request", nargs="*")
    ap.add_argument("--beats", type=int, default=8)
    ap.add_argument("--planner", default=str(ROOT / "adapters" / "mlx-planner2"))
    ap.add_argument("--coder", default=str(ROOT / "adapters" / "mlx-coder2"))
    ap.add_argument("--quality", default="medium",
                    choices=("low", "medium", "high"))
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    req = " ".join(a.request).strip() or input(f"{B}Request:{X} ").strip()
    if not req:
        return 1
    t0 = time.time()

    # -- stage 1 -----------------------------------------------------------
    print(f"\n{B}{C}STAGE 1 · planner{X} {D}({Path(a.planner).name}){X}")
    pm, ptok = load(a.planner)
    beats = plan(pm, ptok, req, stride=6, max_beats=a.beats)
    del pm, ptok
    free()
    if not beats:
        print(f"{R}The planner returned no beats.{X}")
        return 1
    for b in beats:
        secs = f"{b.seconds:g}s" if b.seconds else "?"
        print(f"  {B}{b.n:>2}.{X} [{secs:>4}] {b.intent}")
        if b.narration:
            print(D + textwrap.fill(b.narration, 88,
                                    initial_indent="        ",
                                    subsequent_indent="        ") + X)
    print(f"{D}  {len(beats)} beats, {time.time() - t0:.0f}s{X}")

    # -- stage 2 -----------------------------------------------------------
    print(f"\n{B}{C}STAGE 2 · coder{X} {D}({Path(a.coder).name}){X}")
    cm, ctok = load(a.coder)
    bodies: list[str] = []
    for j, b in enumerate(beats):
        body = ""
        for _ in range(2):
            reply = ask(cm, ctok, CODE_SYSTEM,
                        beat_prompt(req, beats, j, bodies), max_tokens=900)
            cand = extract_code(reply)
            try:
                ast.parse(textwrap.dedent(cand))
                body = cand
                break
            except SyntaxError:
                continue
        bodies.append(body)
        print(f"\n  {B}beat {b.n}: {b.intent}{X}"
              + ("" if body else f"  {R}(did not parse twice; dropped){X}"))
        for line in body.splitlines():
            print(f"    {D}{line}{X}")

    # -- set-up: build what every beat assumes -----------------------------
    missing = missing_names(beats, bodies)
    if missing and any(x.strip() for x in bodies):
        print(f"\n{B}{C}SET-UP{X} {D}(beats use {', '.join(missing[:6])} "
              f"but none builds them){X}")
        pre = extract_code(ask(cm, ctok, CODE_SYSTEM,
                               prelude_prompt(req, bodies, missing),
                               max_tokens=900))
        try:
            ast.parse(textwrap.dedent(pre))
            first = next(k for k, x in enumerate(bodies) if x.strip())
            trial = list(bodies)
            trial[first] = textwrap.dedent(pre).strip() + "\n" + \
                textwrap.dedent(trial[first])
            if len(missing_names(beats, trial)) < len(missing):
                bodies = trial
                for line in pre.splitlines():
                    print(f"    {D}{line}{X}")
            else:
                print(f"  {Y}set-up did not help; skipped{X}")
        except SyntaxError:
            print(f"  {Y}set-up did not parse; skipped{X}")
    del cm, ctok
    free()

    # -- assemble ----------------------------------------------------------
    print(f"\n{B}{C}ASSEMBLE{X}")
    asm = assemble(beats, bodies)
    dropped = 0
    for _ in range(len(bodies)):
        miss = [p for p in asm.problems if "names no beat" in p]
        if not miss:
            break
        names = [n.strip() for n in miss[0].split(":", 1)[1].split(",")
                 if n.strip()]
        print(f"  {Y}{miss[0]}{X}")
        hit = False
        for j, body in enumerate(bodies):
            if body.strip() and any(re.search(rf"\b{re.escape(n)}\b", body)
                                    for n in names):
                bodies[j] = ""
                dropped += 1
                hit = True
                print(f"  {Y}dropping beat {j + 1} ({beats[j].intent}){X}")
        if not hit:
            break
        asm = assemble(beats, bodies)
    if not asm.ok:
        print(f"  {R}not renderable: {asm.problems[:2]}{X}")
        out = ROOT / "data" / "demo" / "last_scene.py"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(asm.code)
        print(f"  code saved to {out}")
        return 1
    kept = sum(1 for x in bodies if x.strip())
    print(f"  {G}clean{X} — {kept} of {len(beats)} beats"
          + (f" ({dropped} dropped)" if dropped else ""))

    # -- render ------------------------------------------------------------
    print(f"\n{B}{C}RENDER{X} {D}({a.quality} quality){X}")
    from forge.harness import RenderHarness
    h = RenderHarness(python_bin=str(ROOT / ".venv" / "bin" / "python"),
                      cache_dir=str(ROOT / "data" / "frames"), timeout=600,
                      store_video=True)
    res = h.render(asm.code, quality=a.quality, use_cache=False)
    for _ in range(3):
        if res.ok:
            break
        k = failing_beat(asm.code, res.stderr or "")
        if k is None or not bodies[k - 1].strip():
            break
        print(f"  {Y}beat {k} failed at runtime ({res.error_kind.value}); "
              f"dropping it and re-rendering{X}")
        bodies[k - 1] = ""
        trial = assemble(beats, bodies)
        if not trial.ok:
            break
        asm = trial
        res = h.render(asm.code, quality=a.quality, use_cache=False)
    out = ROOT / "data" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_scene.py").write_text(asm.code)
    if not res.ok:
        print(f"  {R}render failed: {res.error_kind.value}{X}")
        print(textwrap.indent((res.stderr or "")[-1200:], "    "))
        print(f"  code saved to {out / 'last_scene.py'}")
        return 1
    video = out / "last.mp4"
    video.write_bytes(Path(res.video_path).read_bytes())
    print(f"  {G}rendered{X} {res.duration_s or 0:.1f}s of animation "
          f"in {res.elapsed_s:.0f}s")
    print(f"\n{B}video:{X} {video}\n{B}code:{X}  {out / 'last_scene.py'}"
          f"\n{D}total {time.time() - t0:.0f}s{X}")
    if not a.no_open:
        subprocess.run(["open", str(video)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
