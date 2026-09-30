"""Self-training at scale: the kit coder writes many arcs at once, each scene is
rendered, salvaged and cut into beat-end frames for the critic.

The Mac draws ~30 self-training scenes an hour (one beat at a time, one
arc at a time, with the renders competing for its CPU). On a Kaggle GPU the
same coder can write one beat for sixty arcs in a single batched call, and
the session's CPU cores render the finished scenes meanwhile. This module is
the part both share, so the Kaggle kernel (kaggle/selfgen/selfgen.py) and a
local test run the same code:

  * ``write_arcs(gen, arcs)`` -- beat j of every arc in one batch, then beat
    j + 1, each beat on top of that arc's own earlier beats, exactly as at
    inference (the names hint in the prompt; a beat drawn with another
    subject's blocks, or only a scaffold, resampled up to three times).
  * ``finish(h, beats, bodies)`` -- prune, assemble, render, salvage the
    failing statement or beat, as the pipeline does.
  * ``frames(h, beats, bodies, out)`` -- one more render that records each
    beat's end, and a frame cut there for the vision critic.

A generator is anything with ``batch(system, users, temps, max_tokens)``
returning one completion per prompt.
"""
from __future__ import annotations

import ast
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Protocol

from forge.app.twostage import (Beat, assemble, beat_prompt, extract_code,
                                parsing_prefix, prune_all, runtime_salvage)


class Generator(Protocol):
    def batch(self, system: str, users: list[str], temps: list[float],
              max_tokens: int) -> list[str]: ...


def _parses(code: str) -> bool:
    try:
        ast.parse(textwrap.dedent(code))
        return True
    except SyntaxError:
        return False


def write_arcs(gen: Generator, arcs: list[tuple[str, str, list[Beat]]], system: str,
               max_tokens: int = 900, tries: int = 4) -> list[list[str]]:
    """Every arc's bodies, beat by beat, batched across arcs."""
    from forge.kit.families import hint, relevant
    bodies: list[list[str]] = [[] for _ in arcs]
    depth = max(len(b) for _, _, b in arcs)
    for j in range(depth):
        live = [i for i, (_, _, b) in enumerate(arcs) if j < len(b)]
        users, texts = {}, {}
        for i in live:
            _, req, beats = arcs[i]
            u = beat_prompt(req, beats, j, bodies[i])
            t = f"{beats[j].intent} {beats[j].narration or ''} {req}"
            if h := hint(t):
                u += "\n" + h
            users[i], texts[i] = u, t
        done: dict[int, str] = {}
        first_parsed: dict[int, str] = {}
        last: dict[int, str] = {}
        todo = list(live)
        for k in range(tries):
            if not todo:
                break
            outs = gen.batch(system, [users[i] for i in todo],
                             [0.0 if k == 0 else 0.7] * len(todo), max_tokens)
            again = []
            for i, out in zip(todo, outs):
                cand = extract_code(out)
                last[i] = cand
                if not _parses(cand):
                    again.append(i)
                    continue
                if relevant(cand, texts[i]) is not False:
                    done[i] = cand
                else:
                    first_parsed.setdefault(i, cand)
                    again.append(i)
            todo = again
        for i in live:
            if i in done:
                body = done[i]
            elif i in first_parsed:
                body = first_parsed[i]
            else:
                head = parsing_prefix(last.get(i, ""))
                body = head if head.strip() and any(
                    isinstance(n, ast.Call) for n in ast.walk(ast.parse(head))) else ""
            bodies[i].append(body)
    return bodies


def finish(h, beats: list[Beat], bodies: list[str]):
    """(ok, bodies) after prune, assemble, render and runtime salvage."""
    bodies, _, _ = prune_all(beats, bodies, kit=True)
    asm = assemble(beats, bodies, kit=True)
    res = h.render(asm.code, quality="low", frames=2) if asm.ok else None
    tried: set[int] = set()
    for _ in range(5):
        if res is None or res.ok:
            break
        fix = runtime_salvage(beats, bodies, asm.code, res.stderr or "",
                              kit=True, tried=tried)
        if fix is None:
            break
        bodies = fix[0]
        asm = assemble(beats, bodies, kit=True)
        res = h.render(asm.code, quality="low", frames=2) if asm.ok else None
    return bool(res and res.ok), bodies


def frames(h, beats: list[Beat], bodies: list[str], out: Path, stem: str,
           ffmpeg: str = "ffmpeg") -> list[int] | None:
    """Render once more, recording each live beat's end; cut a frame there to
    ``out/<stem>_<k>.jpg``. Returns the live beat indices, or None."""
    held = [(b + "\nself.wait(0.5)\n_beat_ends.append(self.renderer.time)")
            if b.strip() else b for b in bodies]
    live = [k for k, b in enumerate(bodies) if b.strip()]
    if not live:
        return None
    held[live[0]] = "_beat_ends = []\n" + held[live[0]]
    held[live[-1]] += "\nprint('BEAT_ENDS', _beat_ends)"
    asm = assemble(beats, held, kit=True)
    if not asm.ok:
        return None
    res = h.render(asm.code, quality="low", frames=2, use_cache=False)
    if not res.ok or not res.video_path:
        return None
    m = re.search(r"BEAT_ENDS \[([^\]]*)\]", res.stdout or "")
    ends = [float(x) for x in m.group(1).split(",")] if m and m.group(1) else []
    if len(ends) != len(live):
        return None
    out.mkdir(parents=True, exist_ok=True)
    for k, t in zip(live, ends):
        subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-ss", f"{max(t - 0.25, 0):.2f}",
                        "-i", res.video_path, "-frames:v", "1", "-vf", "scale=640:-1",
                        str(out / f"{stem}_{k}.jpg")])
    return live if all((out / f"{stem}_{k}.jpg").exists() for k in live) else None


class MLXGenerator:
    """The local stand-in: the pipeline's MLX ask(), one prompt at a time."""

    def __init__(self, adapter: str):
        from forge.app.pipeline import load
        self.model, self.tok = load(adapter)

    def batch(self, system, users, temps, max_tokens):
        from forge.app.pipeline import ask
        return [ask(self.model, self.tok, system, u, max_tokens=max_tokens, temp=t)
                for u, t in zip(users, temps)]


# -- the whole run: chunks of arcs written on the GPU, rendered on the CPUs ----

_TEXT = {"Text", "MathTex", "Tex", "Title", "MarkupText", "Paragraph",
         "BulletedList", "Code", "Integer", "DecimalNumber", "Variable"}


def visual(body: str, mobjects: set[str]) -> bool:
    """scripts/scorecard.py's test, here so the Kaggle bundle needs only
    forge/: a kit block beyond the scaffold, a non-text Mobject, or a plot."""
    from forge.kit.kit import KIT_BLOCKS, KIT_SCAFFOLD
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return False
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return bool(calls & (KIT_BLOCKS - KIT_SCAFFOLD)) \
        or bool((calls & mobjects) - _TEXT - {"NumberPlane", "Axes", "NumberLine",
                                              "ComplexPlane"}) or ".plot(" in body


def _render_one(args):
    """Worker: finish one written arc and cut its frames. Runs in a process."""
    rid, req, beats, bodies, out_dir, python_bin, coder = args
    import json as _json
    import manim
    from forge.app.pipeline import CODE_SYSTEM_KIT_TRAINED
    from forge.harness import RenderHarness
    mob = {n for n in dir(manim) if isinstance(getattr(manim, n), type)
           and issubclass(getattr(manim, n), manim.Mobject)}
    # One cache per arc: workers share out_dir, and one clearing a shared
    # cache pulled a video from under another's ffmpeg.
    cache = Path(out_dir) / "cache" / re.sub(r"[^A-Za-z0-9_-]", "_", rid)[:80]
    h = RenderHarness(python_bin=python_bin, cache_dir=str(cache), timeout=240,
                      store_video=True)
    ok, bodies = finish(h, beats, bodies)
    kept = [j for j, b in enumerate(bodies) if b.strip() and visual(b, mob)] if ok else []
    rows = []
    for j in kept:
        rows.append({"messages": [
            {"role": "system", "content": CODE_SYSTEM_KIT_TRAINED},
            {"role": "user", "content": beat_prompt(req, beats, j, bodies)},
            {"role": "assistant", "content": f"```python\n{bodies[j]}\n```"}],
            "meta": {"id": f"self2:{rid}:{j}", "scene": f"self2:{rid}",
                     "source": "kit-self", "task": "beat", "index": j, "coder": coder},
            "prefix": bodies[:j], "intents": [b.intent for b in beats[: j + 1]],
            "request": req})
    live = None
    if kept:
        # The critic judges the scene as its longest kept row sees it: the
        # prefix up to the last kept beat.
        last = max(kept)
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", rid)[:80]
        live = frames(h, beats[: last + 1], bodies[: last + 1], Path(out_dir) / "frames", stem)
    import shutil
    shutil.rmtree(cache, ignore_errors=True)
    return _json.dumps({"arc": rid, "ok": ok, "beats": len(beats), "visual": len(kept),
                        "rows": rows, "live": live,
                        "intents": [b.intent for b in beats], "request": req})


def run(gen: Generator, arcs: list[tuple[str, str, list[Beat]]], out_dir: Path,
        system: str, python_bin: str, coder: str, chunk: int = 48, workers: int = 4,
        deadline: float | None = None) -> None:
    """Write arcs a chunk at a time and render finished chunks meanwhile.
    Appends to out_dir/scenes.jsonl (one line per arc, rows inside)."""
    import json as _json
    import time as _time
    from concurrent.futures import ProcessPoolExecutor
    out_dir.mkdir(parents=True, exist_ok=True)
    done_f = out_dir / "scenes.jsonl"
    done = {_json.loads(l)["arc"] for l in done_f.open()} if done_f.exists() else set()
    todo = [a for a in arcs if a[0] not in done]
    print(f"{len(done)} arcs done, {len(todo)} to go", flush=True)
    pending = []
    import multiprocessing as _mp
    # spawn, not fork: the parent holds the model (a CUDA context on Kaggle),
    # and forking that is how pools hang. Workers import only this module.
    with ProcessPoolExecutor(workers, mp_context=_mp.get_context("spawn")) as pool, \
            done_f.open("a") as f:
        for s in range(0, len(todo), chunk):
            if deadline and _time.time() > deadline:
                print("deadline: no more writing", flush=True)
                break
            part = todo[s: s + chunk]
            t0 = _time.time()
            bodies = write_arcs(gen, part, system)
            print(f"  wrote {len(part)} arcs in {_time.time() - t0:.0f}s", flush=True)
            pending += [pool.submit(_render_one, (rid, req, beats, b, str(out_dir),
                                                  python_bin, coder))
                        for (rid, req, beats), b in zip(part, bodies)]
            still = []
            for p in pending:
                if p.done():
                    f.write(p.result() + "\n")
                    f.flush()
                else:
                    still.append(p)
            pending = still
        for p in pending:
            f.write(p.result() + "\n")
            f.flush()
