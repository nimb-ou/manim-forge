"""The vision critic's renders on Kaggle CPU sessions (no GPU quota).

The Mac renders ~1 critic scene a minute while it does anything else; the
local judge needs ~5 s a scene. This ships unjudged scenes -- assembled,
the kit embedded, instrumented to print their beat ends -- to N Kaggle CPU
kernels, which render and cut one frame per beat; the Mac downloads the
frames and judges them with the local VLM into critic_local.jsonl.

    ./.venv/bin/python scripts/kaggle_critic.py push --take 700 --shards 4
    ./.venv/bin/python scripts/kaggle_critic.py status
    ./.venv/bin/python scripts/kaggle_critic.py collect --threshold 0.8

It takes scenes from the *end* of the critic's queue, so the local critic
(which walks it from the front) and Kaggle do not overlap.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DS = ROOT / "kaggle" / "manim-forge-critic"
KDIR = ROOT / "kaggle" / "critic_render"
USER = "nimbou"
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")


def judged() -> set[str]:
    out = set()
    for n in ("critic.jsonl", "critic_local.jsonl"):
        f = ROOT / "data" / "kit" / n
        if f.exists():
            out |= {json.loads(l)["scene"] for l in f.open() if l.strip()}
    return out


def push(take: int, shards: int) -> None:
    from critic_kit_scenes import instrument, scenes
    from forge.harness.render import find_scene_classes
    done = judged()
    todo = [(s, r) for s, r in scenes().items() if s not in done][::-1][:take]
    DS.mkdir(parents=True, exist_ok=True)
    n = 0
    with (DS / "scenes.jsonl").open("w") as f:
        for s, r in todo:
            got = instrument(r)
            if got is None:
                continue
            code, live, intents = got
            cls = find_scene_classes(code)
            if not cls:
                continue
            f.write(json.dumps({"i": n, "scene": s, "cls": cls[-1], "code": code,
                                "live": live, "intents": intents,
                                "request": r.get("request", "")[:300]}) + "\n")
            n += 1
    (DS / "dataset-metadata.json").write_text(json.dumps(
        {"title": "manim-forge-critic", "id": f"{USER}/manim-forge-critic",
         "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}))
    print(f"{n} scenes -> {DS / 'scenes.jsonl'}")
    exists = subprocess.run([KAGGLE, "datasets", "status", f"{USER}/manim-forge-critic"],
                            capture_output=True, text=True).returncode == 0
    cmd = ([KAGGLE, "datasets", "version", "-p", str(DS), "-m", "critic scenes", "-q"]
           if exists else [KAGGLE, "datasets", "create", "-p", str(DS), "-q"])
    print(subprocess.run(cmd, capture_output=True, text=True).stdout[-300:])
    subprocess.run([KAGGLE, "datasets", "status", f"{USER}/manim-forge-critic"])
    src = (KDIR / "render.py").read_text()
    for i in range(shards):
        d = ROOT / "kaggle" / f"critic_render_{i}"
        d.mkdir(exist_ok=True)
        (d / "render.py").write_text(src.replace("SHARD, NSHARDS = 0, 1",
                                                 f"SHARD, NSHARDS = {i}, {shards}"))
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{USER}/manim-forge-critic-render-{i}",
            "title": f"Manim Forge critic render {i}", "code_file": "render.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": False, "enable_internet": True,
            "dataset_sources": [f"{USER}/manim-forge-critic"],
            "competition_sources": [], "kernel_sources": []}, indent=1))
    print("dataset pushed; run `push-kernels` once it is ready")


def push_kernels(shards: int) -> None:
    for i in range(shards):
        r = subprocess.run([KAGGLE, "kernels", "push", "-p",
                            str(ROOT / "kaggle" / f"critic_render_{i}")],
                           capture_output=True, text=True)
        print(i, (r.stdout + r.stderr).strip()[-200:])


def status(shards: int) -> None:
    for i in range(shards):
        r = subprocess.run([KAGGLE, "kernels", "status",
                            f"{USER}/manim-forge-critic-render-{i}"],
                           capture_output=True, text=True)
        print(i, (r.stdout + r.stderr).strip()[-160:])


def collect(shards: int, threshold: float) -> None:
    from forge.evaluate.local_vision import LocalJudge
    rows = {json.loads(l)["i"]: json.loads(l) for l in (DS / "scenes.jsonl").open()}
    done = judged()
    j = None
    out = ROOT / "data" / "kit" / "critic_local.jsonl"
    for i in range(shards):
        d = ROOT / "data" / "kit" / "kaggle_frames" / str(i)
        if not (d / "manifest.jsonl").exists():
            d.mkdir(parents=True, exist_ok=True)
            r = subprocess.run([KAGGLE, "kernels", "output",
                                f"{USER}/manim-forge-critic-render-{i}", "-p", str(d)],
                               capture_output=True, text=True)
            if not (d / "manifest.jsonl").exists():
                print(i, "no output yet", (r.stdout + r.stderr)[-160:])
                continue
        n = 0
        for l in (d / "manifest.jsonl").open():
            m = json.loads(l)
            if not m.get("ok") or m["scene"] in done:
                continue
            j = j or LocalJudge()
            r = rows[m["i"]]
            ps = [j.p_yes(r["request"], t, d / "frames" / f"{m['i']}_{k}.jpg")
                  for k, t in zip(r["live"], r["intents"])]
            with out.open("a") as f:
                f.write(json.dumps({"scene": m["scene"],
                                    "verdicts": {k: "YES" if p >= threshold else "NO"
                                                 for k, p in zip(r["live"], ps)},
                                    "p": {k: round(p, 4) for k, p in zip(r["live"], ps)},
                                    "threshold": threshold, "rendered": "kaggle"}) + "\n")
            done.add(m["scene"])
            n += 1
        print(f"shard {i}: judged {n}", flush=True)
        shutil.rmtree(d / "frames", ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "push-kernels", "status", "collect"])
    ap.add_argument("--take", type=int, default=700)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.8)
    a = ap.parse_args()
    os.environ.setdefault("PATH", "")
    {"push": lambda: push(a.take, a.shards), "push-kernels": lambda: push_kernels(a.shards),
     "status": lambda: status(a.shards),
     "collect": lambda: collect(a.shards, a.threshold)}[a.cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
