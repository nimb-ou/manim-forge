"""Render every gold scene at low quality and report which ones survive.

A corpus smoke test. These scenes are hand-written, so they are never checked
by the render gate that filters the scraped corpus -- which means a gold scene
can sit broken in the training set indefinitely, teaching the model whatever
it does wrong.

Reports three numbers per scene: whether it rendered, how long the video
actually is, and how long its beats declared. The third is the one that
matters now that ForgeScene pads beats out: a video much shorter than its
declaration means the padding cap was hit, and a beat that hits the cap has
more to say than to show.

Runs scenes in parallel, one process each, because manim is single-threaded
and this machine is not.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib
import inspect
import json
import os
import pkgutil
import re
import subprocess
import sys
import time
from pathlib import Path

PY = "./.venv/bin/python"
#: A scene that cannot render at 480p15 inside this is not a scene
#: anyone can iterate on, whatever it looks like when it finishes.
TIMEOUT_S = 1800
ENV = {**os.environ, "PATH": "/Library/TeX/texbin:" + os.environ.get("PATH", "")}


def scenes() -> list[tuple[str, str, float]]:
    import forge.gold
    from forge.beats import ForgeScene, storyboard
    out = []
    for m in pkgutil.iter_modules(forge.gold.__path__):
        if m.name == "curriculum":
            continue
        mod = importlib.import_module(f"forge.gold.{m.name}")
        for name, obj in vars(mod).items():
            if (inspect.isclass(obj) and issubclass(obj, ForgeScene)
                    and obj is not ForgeScene
                    and obj.__module__ == mod.__name__):
                declared = sum(b.seconds or 0 for b in storyboard(obj))
                out.append((f"forge/gold/{m.name}.py", name, declared))
    return sorted(out)


def duration(path: Path) -> float:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def render(spec) -> dict:
    module, name, declared = spec
    out = Path("data/checks") / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        r = subprocess.run(
            [PY, "-m", "manim", "render", "-ql", "--disable_caching",
             "--media_dir", str(out), module, name],
            capture_output=True, text=True, env=ENV, timeout=TIMEOUT_S,
            stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        # A timeout is a result about that scene, not a reason to abandon the
        # other nineteen. SphereInCube blew the limit once and took the whole
        # run down with it, losing every verdict already computed.
        return {"scene": name, "module": module, "ok": False,
                "render_s": round(time.time() - t0, 1),
                "declared_s": round(declared, 1), "actual_s": 0.0,
                "error": f"timed out after {TIMEOUT_S}s -- too expensive to render"}
    took = time.time() - t0
    vids = list(out.rglob(f"{name}.mp4"))
    vids = [v for v in vids if "partial" not in str(v)]
    ok = r.returncode == 0 and bool(vids)
    err = ""
    if not ok:
        tail = [l for l in (r.stderr or r.stdout).splitlines() if l.strip()]
        err = " | ".join(tail[-3:])[:300]
    return {"scene": name, "module": module, "ok": ok, "render_s": round(took, 1),
            "declared_s": round(declared, 1),
            "actual_s": round(duration(vids[0]), 1) if vids else 0.0,
            "error": err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None,
                    help="scene class names to check; default is all")
    a = ap.parse_args()

    todo = scenes()
    if a.only:
        want = set(a.only)
        todo = [s for s in todo if s[1] in want]
    print(f"{len(todo)} scenes, {a.workers} at a time\n", flush=True)

    results = []
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(render, todo):
            flag = "ok  " if res["ok"] else "FAIL"
            short = "" if res["declared_s"] == 0 else \
                f'  {res["actual_s"]:>6.1f}s / {res["declared_s"]:>5.0f}s declared'
            print(f'  {flag} {res["scene"]:<22} rendered in '
                  f'{res["render_s"]:>6.1f}s{short}', flush=True)
            if res["error"]:
                print(f'       {res["error"]}', flush=True)
            results.append(res)

    Path("data/checks").mkdir(parents=True, exist_ok=True)
    Path("data/checks/verify_gold.json").write_text(json.dumps(results, indent=2))

    bad = [r for r in results if not r["ok"]]
    short = [r for r in results if r["ok"] and r["declared_s"]
             and r["actual_s"] < r["declared_s"] * 0.8]
    print(f'\n  {len(results) - len(bad)}/{len(results)} rendered')
    if short:
        print(f'  {len(short)} noticeably shorter than declared '
              f'(padding cap hit -> under-animated):')
        for r in short:
            print(f'     {r["scene"]:<22} {r["actual_s"]:>6.1f}s vs '
                  f'{r["declared_s"]:>5.0f}s')
    print("  -> data/checks/verify_gold.json")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
