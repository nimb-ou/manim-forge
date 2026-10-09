"""Stack self-training sheets (data/selfgen/sheets) for grading by eye.

    ./.venv/bin/python scripts/stack_selfgen.py 2 5 7   -> a temp jpg (path printed)
    ./.venv/bin/python scripts/stack_selfgen.py --clean  -> clean rows not yet graded, 3 a page
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "selfgen"


def page(ids: list[int]) -> str:
    rows = {json.loads(x)["i"]: json.loads(x) for x in (D / "rows.jsonl").read_text().splitlines()}
    ims = []
    for i in ids:
        f = D / "sheets" / f"{i:03d}.jpg"
        if not f.exists():
            continue
        im = Image.open(f).convert("RGB")
        im = im.resize((900, int(im.height * 900 / im.width)))
        band = Image.new("RGB", (900, 22), (60, 60, 60))
        ImageDraw.Draw(band).text((6, 4), f"{i}  {rows[i]['request'][:110]}", fill=(255, 255, 0))
        ims += [band, im]
    out = Image.new("RGB", (900, sum(x.height for x in ims)))
    y = 0
    for x in ims:
        out.paste(x, (0, y))
        y += x.height
    path = Path(tempfile.gettempdir()) / f"selfgen_{ids[0]}-{ids[-1]}.jpg"
    out.save(path, quality=85)
    return str(path)


def main() -> int:
    if sys.argv[1:] == ["--clean"]:
        graded = set()
        g = ROOT / "data" / "eye" / "selfgen.json"
        if g.exists():
            graded = {int(k) for k in json.loads(g.read_text()) if k.isdigit()}
        ids = [json.loads(x)["i"] for x in (D / "rows.jsonl").read_text().splitlines()
               if json.loads(x).get("clean") and json.loads(x)["i"] not in graded]
        for k in range(0, len(ids), 3):
            print(page(ids[k:k + 3]))
        return 0
    print(page([int(x) for x in sys.argv[1:]]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
