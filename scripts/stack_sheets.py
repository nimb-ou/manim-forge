"""Stack a run's contact sheets for review by eye, a few to an image.

    ./.venv/bin/python scripts/stack_sheets.py is_os1 06 07 08 09
      -> /tmp/.../stack_is_os1_06-09.jpg (path printed)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    tag, ids = sys.argv[1], sys.argv[2:]
    d = ROOT / "data" / "scorecard" / tag
    ims = []
    for i in ids:
        f = next(iter(sorted(d.glob(f"{i}-*.jpg"))), None)
        if f is None:
            continue
        im = Image.open(f).convert("RGB")
        im = im.resize((900, int(im.height * 900 / im.width)))
        band = Image.new("RGB", (900, 22), (60, 60, 60))
        ImageDraw.Draw(band).text((6, 4), f"{i}  {f.stem[3:]}", fill=(255, 255, 0))
        ims += [band, im]
    if not ims:
        print("no sheets")
        return 1
    out = Image.new("RGB", (900, sum(x.height for x in ims)))
    y = 0
    for x in ims:
        out.paste(x, (0, y))
        y += x.height
    path = Path(tempfile.gettempdir()) / f"stack_{tag}_{ids[0]}-{ids[-1]}.jpg"
    out.save(path, quality=88)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
