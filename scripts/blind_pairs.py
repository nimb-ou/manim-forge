"""Two runs' contact sheets side by side, order hidden, for grading by eye.

    ./.venv/bin/python scripts/blind_pairs.py world_tuned world_v15
      -> temp jpgs, two prompts a page: for each prompt the sheet marked X
         and the sheet marked Y; which run is which is written to
         data/eye/<a>__<b>.key.json, to be opened only after grading.

The order is drawn per prompt from a fixed seed, so a rerun shows the same
pages. Grades go in data/eye/<a>.json and <b>.json after unblinding.
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def sheet(tag: str, i: int):
    f = next(iter(sorted((ROOT / "data" / "scorecard" / tag).glob(f"{i:02d}-*.jpg"))), None)
    if f is None:
        im = Image.new("RGB", (900, 120), (40, 0, 0))
        ImageDraw.Draw(im).text((10, 50), "did not render", fill=(255, 120, 120))
        return im, None
    im = Image.open(f).convert("RGB")
    return im.resize((900, int(im.height * 900 / im.width))), f.stem[3:]


def main() -> int:
    a, b = sys.argv[1], sys.argv[2]
    ids = sorted({int(p.name[:2]) for t in (a, b)
                  for p in (ROOT / "data" / "scorecard" / t).glob("[0-9][0-9]-*.py")})
    rng = random.Random(f"{a}|{b}")
    key, ims = {}, []
    for i in ids:
        first_a = rng.random() < 0.5
        key[f"{i:02d}"] = {"X": a if first_a else b, "Y": b if first_a else a}
        for lab in ("X", "Y"):
            im, slug = sheet(key[f"{i:02d}"][lab], i)
            band = Image.new("RGB", (900, 22), (60, 60, 60))
            ImageDraw.Draw(band).text((6, 4), f"{i:02d}{lab}  {slug or ''}", fill=(255, 255, 0))
            ims.append((i, band, im))
    (ROOT / "data" / "eye" / f"{a}__{b}.key.json").write_text(json.dumps(key, indent=1))
    pages = {}
    for i, band, im in ims:
        pages.setdefault((ids.index(i)) // 2, []).extend([band, im])
    for n, parts in sorted(pages.items()):
        out = Image.new("RGB", (900, sum(p.height for p in parts)))
        y = 0
        for p in parts:
            out.paste(p, (0, y))
            y += p.height
        path = Path(tempfile.gettempdir()) / f"blind_{a}_{b}_{n:02d}.jpg"
        out.save(path, quality=85)
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
