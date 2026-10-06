"""A scene-level vision judge: does the whole video answer the request, correctly?

judge_sheets.py asks, beat by beat, whether a frame shows its *own* intent.
That cannot see three failures the sheets keep showing: a scene whose beats
each match their intents while the whole answers a different question; a
number on screen that is wrong (the 15%-off jacket that "costs £72.80" after
VAT nobody asked about); and beats that do not build on each other. When the
model writes its own intents (forge/app/oneshot.py) it could also write easy
ones. So this asks four yes/no questions about the scene as a whole:

  ANSWERS   the video reaches the answer, or shows the idea asked for
  CORRECT   every number, label and equation on screen is right for *this* request
  COHERENT  one connected explanation, later frames building on earlier ones
  CLEAN     nothing overlaps, is cut off, or is unreadable

A scene is GOOD when the first three are YES. A failed render is not good.

    ./.venv/bin/python scripts/judge_scenes.py is_p3_k8 is_os_base
      -> data/scorecard/<tag>/scene_judge.json ; prints the shares per tag
"""
from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from judge_sheets import MODELS, ask, tiles  # noqa: E402

KEYS = ("ANSWERS", "CORRECT", "COHERENT", "CLEAN")
ASK = """You are judging a short animated maths explanation made for this request:
"{request}"

The images are the last frame of each beat, in order. The beats were meant to show:
{beats}

Work out the right answer to the request yourself first and compare it with what the frames show. A problem you noticed in your working means NO. Then answer each question YES or NO:
ANSWERS: does the video as a whole answer the request -- reach the answer, or show the idea that was asked for?
CORRECT: is every number, label and equation visible correct for THIS request? Be strict: NO if any value is wrong, if the final answer shown differs from the right answer, or if the video adds steps the request did not ask for (a different problem's numbers, an extra tax, a different shape).
COHERENT: do the frames build one connected explanation, later frames building on earlier ones, nothing unrelated?
CLEAN: is everything readable, with nothing overlapping or cut off?

Reply with your short working, then exactly four lines:
ANSWERS: YES or NO
CORRECT: YES or NO
COHERENT: YES or NO
CLEAN: YES or NO"""


def judge(tag: str, key: str) -> dict:
    d = ROOT / "data" / "scorecard" / tag
    rows = json.loads((d / "scorecard.json").read_text())["rows"]
    out_path = d / "scene_judge.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    for i, r in enumerate(rows, 1):
        k = f"{i:02d}"
        if k in done or not r["ok"]:
            continue
        py = next(iter(sorted(d.glob(f"{k}-*.py"))), None)
        sheet = next(iter(sorted(d.glob(f"{k}-*.jpg"))), None)
        if py is None or sheet is None:
            continue
        intents = re.findall(r"^\s*# beat \d+:\s*(.*)$", py.read_text(), re.M)
        imgs = tiles(sheet, len(intents))
        if not intents or len(imgs) != len(intents):
            continue
        text = ASK.format(request=r["title"], beats="\n".join(
            f"{j + 1}: {t}" for j, t in enumerate(intents)))
        parts = [{"text": text}] + [{"inline_data": {"mime_type": "image/jpeg",
                                                     "data": base64.b64encode(b).decode()}}
                                    for b in imgs]
        got = {}
        for _ in range(3):
            reply = ask(key, parts)
            if reply is None:
                break
            got = {m.group(1).upper(): m.group(2).upper() for m in re.finditer(
                r"^\W*(ANSWERS|CORRECT|COHERENT|CLEAN)\W*:\W*(YES|NO)", reply, re.M | re.I)}
            if all(x in got for x in KEYS):
                break
        if not all(x in got for x in KEYS):
            print(f"    {tag} {k}: incomplete reply, not recorded", flush=True)
            continue
        done[k] = {"title": r["title"], "models": list(MODELS), **got,
                   "working": (reply or "")[:600]}
        out_path.write_text(json.dumps(done, indent=1))
        print(f"  {tag} {k}: " + " ".join(f"{x[:4]}={got[x][0]}" for x in KEYS)
              + f" ({r['title'][:40]})", flush=True)
    n = len(rows)
    share = {x.lower(): round(sum(v[x] == "YES" for v in done.values()) / n, 3)
             for x in KEYS}
    good = sum(all(v[x] == "YES" for x in KEYS[:3]) for v in done.values())
    judged = len(done) + sum(1 for r in rows if not r["ok"])
    return {"tag": tag, "good": good, "n": n, "good_share": round(good / n, 3),
            **share, "scenes_judged": judged}


def main() -> int:
    from forge.synth.teacher import Teacher
    key = Teacher(provider="gemini", model=MODELS[0])._key
    args = sys.argv[1:]
    if "--model" in args:
        i = args.index("--model")
        MODELS[:] = [args[i + 1]]
        del args[i: i + 2]
    for tag in args:
        s = judge(tag, key)
        print(f"{tag:24s} good {s['good']}/{s['n']} ({s['good_share']:.0%})  answers "
              f"{s['answers']:.0%} correct {s['correct']:.0%} coherent "
              f"{s['coherent']:.0%} clean {s['clean']:.0%}  "
              f"({s['scenes_judged']}/{s['n']} judged)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
