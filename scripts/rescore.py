"""Re-score saved scorecard scenes: visual, new, *and about the beat*.

GRPO v2 scored 76% visual-and-new and its contact sheets were a plane with
one vector for a neural network, for Bayes' theorem, for backpropagation.
The scorecard counted any non-scaffold block as a picture; it did not ask
whether the picture belonged to the beat. This adds the relevance test the
training data already passes through (filter_kit_beats.FAMILIES: a kit
block counts only if its family's words appear in the beat's intent or the
request). Beats drawn with raw Manim only are not judged, as in the filter.

The denominator is every *planned* beat, and a scene that failed to render
scores zero, so dropping beats cannot raise the number.

    ./.venv/bin/python scripts/rescore.py short_kit_c5 short_kit_grpo2 ...
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from forge.kit.families import relevant as _relevant  # noqa: E402
from forge.app.twostage import repeats_earlier  # noqa: E402
from scorecard import visual  # noqa: E402


def beats_of(code: str) -> list[tuple[str, str]]:
    """(intent, body) for each beat left in a saved scene's construct()."""
    src = code[code.index("def construct(self)"):] if "def construct(self)" in code else code
    parts = re.split(r"^\s*# beat \d+:\s*(.*)$", src, flags=re.M)
    out = []
    for k in range(1, len(parts), 2):
        body = "\n".join(l[8:] if l.startswith(" " * 8) else l.strip()
                         for l in parts[k + 1].splitlines())
        out.append((parts[k].strip(), body))
    return out


def relevant(body: str, text: str) -> bool:
    return _relevant(body, text) is not False


def score(tag: str, mob: set[str]) -> dict:
    d = ROOT / "data" / "scorecard" / tag
    rows = json.loads((d / "scorecard.json").read_text())["rows"]
    planned = good = vis = 0
    per = []
    for i, r in enumerate(rows, 1):
        planned += r["beats"]
        if not r["ok"]:
            per.append((i, 0, 0, r["beats"]))
            continue
        f = next(iter(sorted(d.glob(f"{i:02d}-*.py"))), None)
        if f is None:
            continue
        beats = beats_of(f.read_text())
        bodies = [b for _, b in beats]
        v = g = 0
        for k, (intent, body) in enumerate(beats):
            if visual(body, mob) and not repeats_earlier(body, bodies[:k]):
                v += 1
                if relevant(body, (intent + " " + r["title"]).lower()):
                    g += 1
        vis += v
        good += g
        per.append((i, v, g, r["beats"]))
    return {"tag": tag, "planned": planned, "visual_new": vis,
            "visual_new_relevant": good,
            "share_visual_new": round(vis / planned, 3),
            "share_relevant": round(good / planned, 3), "per": per}


def main() -> int:
    import manim
    mob = {n for n in dir(manim) if isinstance(getattr(manim, n), type)
           and issubclass(getattr(manim, n), manim.Mobject)}
    for tag in [a for a in sys.argv[1:] if not a.startswith("-")]:
        s = score(tag, mob)
        print(f"{tag:22s} planned={s['planned']:3d}  visual+new={s['share_visual_new']:.0%}"
              f"  +relevant={s['share_relevant']:.0%}")
        if "-v" in sys.argv:
            print("   ", " ".join(f"{i}:{g}/{n}" for i, v, g, n in s["per"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
