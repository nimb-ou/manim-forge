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


def already_done() -> set[str]:
    if not LEDGER.exists():
        return set()
    out = set()
    for line in LEDGER.open():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("ok"):
            out.add(rec["scene"])
    return out


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
    done = already_done()
    todo = [s for s in scenes() if s[1] not in done]
    print(f"{len(scenes())} scenes | {len(done)} already rendered at "
          f"presentation quality | {len(todo)} to go", flush=True)

    with LEDGER.open("a") as ledger:
        for module, name, declared in todo:
            print(f"\n  {name}  ({declared:.0f}s declared)", flush=True)
            t0 = time.time()
            try:
                r = subprocess.run(
                    [PY, "-m", "manim", "render", QUALITY[a.quality],
                     "--disable_caching",
                     "--save_sections", "--media_dir", str(OUT), module, name],
                    capture_output=True, text=True, env=ENV,
                    timeout=a.timeout, stdin=subprocess.DEVNULL, cwd=ROOT)
                ok = r.returncode == 0
                err = "" if ok else " | ".join(
                    (r.stderr or r.stdout).strip().splitlines()[-2:])[:240]
            except subprocess.TimeoutExpired:
                ok, err = False, f"timed out after {a.timeout}s"
            took = time.time() - t0
            ledger.write(json.dumps({
                "scene": name, "module": module, "quality": a.quality,
                "ok": ok, "seconds": round(took, 1),
                "declared_s": declared, "error": err}) + "\n")
            ledger.flush()
            print(f"    {'ok' if ok else 'FAILED'} in {took/60:.1f} min"
                  + (f"\n    {err}" if err else ""), flush=True)

    vids = list(OUT.rglob("*.mp4"))
    print(f"\n{len(vids)} video files under {OUT}", flush=True)


if __name__ == "__main__":
    main()
