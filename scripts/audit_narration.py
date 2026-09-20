"""Audit every gold scene's narration track.

The narration is the half of a scene that no render can check. A wrong number
on screen is caught by the primitives; a wrong number *spoken* renders
perfectly and is only found by reading. That has now happened three times --
invented triangle angles, a path called three steps that was four, and stray
non-Latin characters that would have been read aloud by a speech engine.

So the checks here are the ones that can be made mechanical:

  charset   every character must be speakable Latin text
  empty     every beat must carry narration; a silent beat is an unfinished one
  pace      words per second must be sayable within the beat's own duration
  digits    bare numerals belong in the picture, not the script -- a narrator
            reads "23", a writer means "twenty-three", and the mismatch is
            where invented numbers hide

Nothing here can tell whether a spoken number is *true*; that needs the scene
to compute it. What it can do is make sure a number in the script is written
out, which forces the author to look at it.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import re
import sys
from pathlib import Path

ALLOWED_EXTRA = set(" \t\n—–’‘“”…'\"()[],.;:!?%/+-=*°×·")
#: Must match ForgeScene.NARRATION_WPS -- the audit and the runtime
#: padding have to agree, or the audit flags beats the renderer would
#: have fixed and passes ones it then stretches.
from forge.beats import ForgeScene

WPS = ForgeScene.NARRATION_WPS


def _round_up(x: float) -> int:
    return int(x) + 1 if x > int(x) else int(x)


def scene_classes():
    import forge.gold
    from forge.beats import ForgeScene
    out = []
    for m in pkgutil.iter_modules(forge.gold.__path__):
        if m.name in ("curriculum",):
            continue
        mod = importlib.import_module(f"forge.gold.{m.name}")
        for name, obj in vars(mod).items():
            if (inspect.isclass(obj) and issubclass(obj, ForgeScene)
                    and obj is not ForgeScene and obj.__module__ == mod.__name__):
                out.append((m.name, name, obj))
    return sorted(out)


def check(module, name, cls):
    from forge.beats import storyboard
    problems = []
    for b in storyboard(cls):
        where = f"{module}.{name}.{b.name}"
        text = (b.narration or "").strip()

        if not text:
            problems.append((where, "empty", "beat has no narration"))
            continue

        bad = sorted({c for c in text
                      if not (c.isalpha() and c.isascii()) and c not in ALLOWED_EXTRA})
        if bad:
            problems.append((where, "charset",
                             "unspeakable characters: " + " ".join(
                                 f"{c!r}(U+{ord(c):04X})" for c in bad)))

        digits = re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", text)
        if digits:
            problems.append((where, "digits",
                             "numerals in script, write them out: "
                             + ", ".join(digits)))

        words = len(text.split())
        needed = words / WPS
        if needed > b.seconds + 0.5:
            problems.append((where, "pace",
                             f"{words} words need {needed:.0f}s at {WPS} w/s, "
                             f"beat declares {b.seconds:.0f}s "
                             f"-> declare {_round_up(needed)}s"))
        elif b.seconds > needed * 2.2 and b.seconds - needed > 4:
            problems.append((where, "pace",
                             f"{words} words need only {needed:.0f}s but the "
                             f"beat declares {b.seconds:.0f}s -- mostly silence"))
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on any problem")
    a = ap.parse_args()

    all_problems = []
    scenes = scene_classes()
    for module, name, cls in scenes:
        all_problems += check(module, name, cls)

    by_kind: dict[str, int] = {}
    for _, kind, _ in all_problems:
        by_kind[kind] = by_kind.get(kind, 0) + 1

    print(f"{len(scenes)} gold scenes audited\n")
    if not all_problems:
        print("  clean")
        return 0
    for where, kind, msg in all_problems:
        print(f"  {kind:<8} {where}")
        print(f"           {msg}")
    print("\n  " + "  ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))
    return 1 if a.strict else 0


if __name__ == "__main__":
    sys.exit(main())
