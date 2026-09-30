"""A vision judge for scorecard runs: does each beat's last frame show its idea?

The scorecard's relevance share counts a beat as a picture if its code calls
a non-scaffold block whose subject words match the beat -- a bare square or
a lone arrow passes, and by eye the shipped configuration was ~43% where the
metric said 63%. This asks a vision model instead, beat by beat, with the
frame the scene shows at the end of that beat (the pipeline's beat marks,
scorecard contact sheets since 2026-09-27) and the beat's intent.

A beat is YES only if its frame shows a picture that illustrates the
beat's idea. The share is YES beats over *planned* beats (a failed render
or a dropped beat scores zero), comparable to rescore.py's.

    ./.venv/bin/python scripts/judge_sheets.py ab_short_names ab_held_names
      -> data/scorecard/<tag>/judge.json ; prints the share per tag
"""
from __future__ import annotations

import base64
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

URL = "https://generativelanguage.googleapis.com/v1beta"
# Each model has its own daily quota; rotated on 429s (2026-09-29: these
# answered when flash-latest and flash-lite-latest were spent).
MODELS = ["gemini-flash-latest", "gemini-3.6-flash", "gemini-3.5-flash",
          "gemini-3-flash-preview", "gemini-flash-lite-latest", "gemini-3.1-flash-lite"]
TILE = (480, 270)

ASK = """You are judging frames from a short animated maths explanation of:
"{request}"

Frame k is the last frame of beat k. For each beat, answer YES only if its
frame shows a PICTURE that illustrates that beat's idea -- a diagram, graph,
geometric figure, chart or arrangement that a viewer would learn the idea
from. Answer NO for: only text or an equation; empty axes, a bare grid or a
bare number line; a lone shape or arrow that does not show the idea; a
picture of something else; a cluttered or overlapping mess.

{beats}

Answer one line per beat, exactly "k: YES" or "k: NO"."""


def tiles(sheet: Path, n: int) -> list[bytes]:
    from PIL import Image
    im = Image.open(sheet)
    cols = im.size[0] // TILE[0]
    out = []
    for k in range(n):
        r, c = divmod(k, cols)
        box = (c * TILE[0], r * TILE[1], (c + 1) * TILE[0], (r + 1) * TILE[1])
        if box[3] > im.size[1]:
            break
        buf = io.BytesIO()
        im.crop(box).convert("RGB").save(buf, "JPEG", quality=85)
        out.append(buf.getvalue())
    return out


def ask(key: str, parts: list) -> str | None:
    body = json.dumps({"contents": [{"role": "user", "parts": parts}],
                       "generationConfig": {"temperature": 0,
                                            "maxOutputTokens": 2048}}).encode()
    for attempt in range(8):
        model = MODELS[attempt % len(MODELS)]
        req = urllib.request.Request(f"{URL}/models/{model}:generateContent",
                                     data=body, headers={"x-goog-api-key": key,
                                                         "Content-Type": "application/json"})
        try:
            data = json.load(urllib.request.urlopen(req, timeout=120))
            return "".join(p.get("text", "") for p in
                           data["candidates"][0]["content"]["parts"])
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503):
                time.sleep(20 * (attempt + 1))
                continue
            print(f"    HTTP {e.code}: {e.read()[:160]!r}", flush=True)
            return None
        except (urllib.error.URLError, TimeoutError, KeyError):
            time.sleep(10)
    return None


def judge(tag: str, key: str) -> dict:
    d = ROOT / "data" / "scorecard" / tag
    rows = json.loads((d / "scorecard.json").read_text())["rows"]
    out_path = d / "judge.json"
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
        # Thinking models spend output tokens before answering; a reply cut
        # short once scored the missing beats as NO (2026-09-30: 13% where
        # the same runs had scored ~45%). Only a verdict for every beat counts.
        verdicts = {}
        for _ in range(3):
            reply = ask(key, parts)
            if reply is None:
                break
            verdicts = {int(m.group(1)): m.group(2).upper() for m in
                        re.finditer(r"^\s*(\d+)\s*[:.)-]\s*(YES|NO)", reply, re.M | re.I)
                        if 0 < int(m.group(1)) <= len(intents)}
            if len(verdicts) == len(intents):
                break
        if len(verdicts) != len(intents):
            print(f"    {tag} {k}: incomplete reply, not recorded", flush=True)
            continue
        done[k] = {"title": r["title"], "intents": intents,
                   "verdicts": {str(a): b for a, b in sorted(verdicts.items())}}
        out_path.write_text(json.dumps(done, indent=1))
        yes = sum(v == "YES" for v in verdicts.values())
        print(f"  {tag} {k}: {yes}/{r['beats']} ({r['title'][:40]})", flush=True)
    planned = sum(r["beats"] for r in rows)
    yes = sum(sum(v == "YES" for v in s["verdicts"].values()) for s in done.values())
    judged = len(done) + sum(1 for r in rows if not r["ok"])
    return {"tag": tag, "planned": planned, "yes": yes, "share": round(yes / planned, 3),
            "scenes_judged": judged, "of": len(rows)}


def main() -> int:
    from forge.synth.teacher import Teacher
    key = Teacher(provider="gemini", model=MODELS[0])._key
    args = sys.argv[1:]
    # --model M: judge every scene with one model, so runs compared with each
    # other are judged alike (the rotation lands on whichever has quota).
    if "--model" in args:
        i = args.index("--model")
        MODELS[:] = [args[i + 1]]
        del args[i: i + 2]
    for tag in args:
        s = judge(tag, key)
        print(f"{tag:24s} judge share {s['share']:.0%}  ({s['yes']}/{s['planned']} planned beats; "
              f"{s['scenes_judged']}/{s['of']} scenes judged)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
