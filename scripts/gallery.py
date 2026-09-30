"""A gallery of real outputs for the release: scenes from a scorecard run,
re-rendered at medium quality, on one page with each scene's beats.

Picks the scenes the vision judge passed most beats of (judge.json, else
judge_local.json), or the ones named with --pick.

    ./.venv/bin/python scripts/gallery.py v8_held --n 10
    ./.venv/bin/python scripts/gallery.py ab_held_scaf --pick 03,13,15
      -> data/gallery/<tag>/index.html and NN.mp4
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--pick", default="")
    ap.add_argument("--quality", default="medium")
    a = ap.parse_args()
    d = ROOT / "data" / "scorecard" / a.tag
    j = next((json.loads((d / n).read_text()) for n in ("judge.json", "judge_local.json")
              if (d / n).exists()), {})
    if a.pick:
        keys = [f"{int(k):02d}" for k in a.pick.split(",")]
    else:
        score = {k: sum(v == "YES" for v in s["verdicts"].values()) / max(len(s["verdicts"]), 1)
                 for k, s in j.items()}
        keys = sorted(score, key=lambda k: -score[k])[: a.n]
    from forge.harness import RenderHarness
    h = RenderHarness(python_bin=str(ROOT / ".venv" / "bin" / "python"),
                      cache_dir=str(ROOT / "data" / "frames"), timeout=900, store_video=True)
    out = ROOT / "data" / "gallery" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    cards = []
    for k in keys:
        py = next(iter(sorted(d.glob(f"{k}-*.py"))), None)
        if py is None:
            continue
        code = py.read_text()
        res = h.render(code, quality=a.quality, use_cache=False)
        if not res.ok or not res.video_path:
            print(f"  {k}: render failed", flush=True)
            continue
        shutil.copy(res.video_path, out / f"{k}.mp4")
        intents = re.findall(r"^\s*# beat \d+:\s*(.*)$", code, re.M)
        title = j.get(k, {}).get("title") or py.stem.split("-", 1)[-1].replace("-", " ")
        verdicts = j.get(k, {}).get("verdicts", {})
        beats = "".join(
            f"<li class='{'yes' if verdicts.get(str(n + 1)) == 'YES' else 'no'}'>"
            f"{html.escape(t)}</li>" for n, t in enumerate(intents))
        cards.append(f"<article><video src='{k}.mp4' controls preload='metadata'></video>"
                     f"<h2>{html.escape(title)}</h2><ol>{beats}</ol></article>")
        print(f"  {k}: {title[:60]}", flush=True)
    (out / "index.html").write_text(f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Manim Forge gallery</title>
<style>
:root{{--bg:#0f1115;--fg:#e8e6e1;--mute:#8a8f98;--yes:#7fc97f;--no:#d9776a}}
body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:15px/1.5 Georgia,serif}}
h1{{font-weight:normal;margin:0 0 4px}} p{{color:var(--mute);margin:0 0 24px;max-width:65ch}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:24px}}
video{{width:100%;background:#000;border-radius:4px}} h2{{font-size:17px;margin:8px 0 4px}}
ol{{margin:0;padding-left:20px;font-size:13px;color:var(--mute)}} li.yes::marker{{color:var(--yes)}}
li.no::marker{{color:var(--no)}}
</style>
<h1>Manim Forge</h1>
<p>Real outputs from run <code>{html.escape(a.tag)}</code>: one sentence in, a planned,
rendered animation out, on a Mac. Beat numbers are green where the vision judge
said the frame shows the beat's idea, red where it did not.</p>
<main>{''.join(cards)}</main>
""")
    print(f"{len(cards)} scenes -> {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
