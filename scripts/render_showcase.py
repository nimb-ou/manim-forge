"""Render every gold scene at presentation quality, resumably.

These are the reference renders: what the product is supposed to look like,
what goes on the site's example gallery, and what a human judges the style
against. They are also, conveniently, hours of pure CPU work with no API
involvement -- which is exactly what is needed for the ten hours a day the
Gemini quota is gone.

Resumable by output file, so it can be killed and restarted freely. Renders
the cheapest-first so the gallery fills early rather than blocking on the one
3-D scene that takes half an hour.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import pkgutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge.harness.render import _killpg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PY = "./.venv/bin/python"
ENV = {**os.environ, "PATH": "/Library/TeX/texbin:" + os.environ.get("PATH", ""),
       "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT)}
OUT = ROOT / "data" / "showcase"
LEDGER = OUT / "rendered.jsonl"

QUALITY = {"low": "-ql", "medium": "-qm", "high": "-qh", "4k": "-qk"}


def scenes() -> list[tuple[str, str, float]]:
    import forge.gold
    from forge.beats import ForgeScene, storyboard
    found = []
    for m in pkgutil.iter_modules(forge.gold.__path__):
        if m.name == "curriculum":
            continue
        mod = importlib.import_module(f"forge.gold.{m.name}")
        for name, obj in vars(mod).items():
            if (inspect.isclass(obj) and issubclass(obj, ForgeScene)
                    and obj is not ForgeScene
                    and obj.__module__ == mod.__name__):
                declared = sum(b.seconds or 0 for b in storyboard(obj))
                found.append((f"forge/gold/{m.name}.py", name, declared))
    # Cheapest first: the gallery fills early instead of blocking on the one
    # 3-D scene that takes half an hour.
    return sorted(found, key=lambda t: t[2])


def already_done(quality: str, timeout: int) -> set[str]:
    """Scenes not worth attempting again this pass.

    Rendered ones, obviously -- and also ones that have already exhausted a
    timeout at least as long as this one. SphereInCube is a 3-D scene with a
    600-point cloud; at 1080p60 it ran the full 90 minutes and was killed,
    and the pool then restarted the job every half hour to spend another 90
    minutes discovering the same thing. Retrying an identical attempt is not
    resumption, it is a loop.
    """
    if not LEDGER.exists():
        return set()
    out, hopeless = set(), set()
    for line in LEDGER.open():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("ok"):
            out.add(rec["scene"])
        elif (not rec.get("interrupted")
              and "timed out" in (rec.get("error") or "")
              and rec.get("quality") == quality
              # Records written before this field existed ran under the
              # then-default of 5400s. Reading them as timeout=0 made
              # "already failed at a clock this long" always false, so the
              # skip never fired and SphereInCube was retried anyway.
              and rec.get("timeout", 5400) >= timeout):
            hopeless.add(rec["scene"])
    for s in hopeless - out:
        print(f"  skipping {s}: already timed out at >={timeout}s, "
              f"{quality} quality — raise --timeout to retry", flush=True)
    return out | hopeless


def main() -> None:
    ap = argparse.ArgumentParser()
    # Named, not "-qh". argparse reads any value beginning with a dash as
    # another option, so --quality -qh fails with "expected one argument" --
    # which is exactly how this crash-looped four times before the supervisor
    # gave up on it.
    ap.add_argument("--quality", default="high",
                    choices=list(QUALITY))
    ap.add_argument("--timeout", type=int, default=5400)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    done = already_done(a.quality, a.timeout)
    todo = [s for s in scenes() if s[1] not in done]
    print(f"{len(scenes())} scenes | {len(done)} already rendered at "
          f"presentation quality | {len(todo)} to go", flush=True)

    with LEDGER.open("a") as ledger:
        for module, name, declared in todo:
            print(f"\n  {name}  ({declared:.0f}s declared)", flush=True)
            t0 = time.time()
            proc = subprocess.Popen(
                [PY, "-m", "manim", "render", QUALITY[a.quality],
                 "--disable_caching",
                 "--save_sections", "--media_dir", str(OUT), module, name],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env=ENV, stdin=subprocess.DEVNULL, cwd=ROOT,
                start_new_session=True)
            try:
                out, _ = proc.communicate(timeout=a.timeout)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                _killpg(proc)
                out, _ = proc.communicate()
                rc = None
            took = time.time() - t0

            if rc is not None and rc < 0:
                # Killed by a signal, which here means the supervisor stopped
                # us -- not a verdict on the scene. Writing "ok": false for
                # this is how the ledger came to hold five failed scenes, four
                # of which had already rendered successfully and one of which
                # had simply never been allowed to finish. An interrupted
                # attempt records nothing and is retried next run.
                print(f"    interrupted by signal {-rc} after "
                      f"{took/60:.1f} min -- not recorded", flush=True)
                raise SystemExit(128 - rc)

            ok = rc == 0
            if rc is None:
                err = f"timed out after {a.timeout}s"
            else:
                err = "" if ok else " | ".join(
                    (out or "").strip().splitlines()[-2:])[:240]
            ledger.write(json.dumps({
                "scene": name, "module": module, "quality": a.quality,
                "ok": ok, "seconds": round(took, 1),
                "declared_s": declared, "timeout": a.timeout,
                "error": err}) + "\n")
            ledger.flush()
            print(f"    {'ok' if ok else 'FAILED'} in {took/60:.1f} min"
                  + (f"\n    {err}" if err else ""), flush=True)

    vids = list(OUT.rglob("*.mp4"))
    print(f"\n{len(vids)} video files under {OUT}", flush=True)


if __name__ == "__main__":
    main()
