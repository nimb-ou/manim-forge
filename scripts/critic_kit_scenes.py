"""A visual critic for the kit teacher's scenes: does the picture show the beat?

The step that lifts quality in the systems that work (Code2Video's Critic,
TheoremExplainAgent's visual checks) is a vision model looking at rendered
frames. The keyword filter can only ask whether a block's family matches
the beat's words; it passed pictures that are about the right subject and
still show nothing. This renders each teacher scene, samples one frame at
the end of every beat, and asks a vision model, beat by beat, whether that
frame shows the beat's idea as a picture. filter_kit_beats.py drops beats
judged NO.

Gemini's free tier runs out daily; on a 429 this waits and carries on.

    ./.venv/bin/python -u scripts/critic_kit_scenes.py
      -> data/kit/critic.jsonl  {scene, verdicts: {index: "YES"|"NO"}}
"""
from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forge.app.twostage import Beat, assemble, extract_code  # noqa: E402

SRC = ROOT / "data" / "kit" / "kit_beats.jsonl"
OUT = ROOT / "data" / "kit" / "critic.jsonl"
# gemini-3.7-flash ran out of quota and the critic sat on 429s for days;
# these answer (2026-09-28). Tried in turn on 429s.
MODEL = "gemini-flash-latest"
# Each model has its own daily quota; rotated on 429s (2026-09-29: these
# answered when flash-latest and flash-lite-latest were spent).
MODELS = ["gemini-flash-latest", "gemini-3.6-flash", "gemini-3.5-flash",
          "gemini-3-flash-preview", "gemini-flash-lite-latest", "gemini-3.1-flash-lite"]
SELF = ROOT / "data" / "kit" / "self_beats.jsonl"
URL = "https://generativelanguage.googleapis.com/v1beta"

ASK = """You are judging frames from a short animated maths explanation of:
"{request}"

Frame k is the last frame of beat k. For each beat, answer YES only if its
frame shows a PICTURE that illustrates that beat's idea -- a diagram, graph,
geometric figure, chart or arrangement that a viewer would learn the idea
from. Answer NO for: only text or an equation; empty axes, a bare grid or a
bare number line; a lone shape, dot or arrow that does not show the idea; a
picture of something else (vectors standing in for a graph, a plane for a
network); a cluttered or overlapping mess.

{beats}

Answer one line per beat, exactly "k: YES" or "k: NO"."""


def scenes() -> dict[str, dict]:
    """For each scene, its longest rendered prefix: bodies and intents."""
    best: dict[str, dict] = {}
    # Self-training scenes first: the model's own output is the riskier data
    # (a shaded block through the title passed every other filter), and
    # Gemini's rate limits mean not everything is judged before an SFT.
    lines = (SELF.read_text().splitlines() if SELF.exists() else []) + \
        SRC.read_text().splitlines()
    for l in lines:
        if not l.strip():
            continue
        r = json.loads(l)
        if "prefix" not in r:
            continue
        s = r["meta"]["scene"]
        if s not in best or r["meta"]["index"] > best[s]["meta"]["index"]:
            best[s] = r
    return best


def main() -> int:
    dry = "--dry" in sys.argv         # render and cut frames, skip the API
    from forge.synth.teacher import Teacher
    key = Teacher(provider="gemini", model=MODEL)._key
    from forge.harness import RenderHarness
    h = RenderHarness(python_bin=str(ROOT / ".venv" / "bin" / "python"),
                      cache_dir=str(ROOT / "data" / "frames"), timeout=300,
                      store_video=True)
    done = {json.loads(l)["scene"] for l in OUT.open() if l.strip()} \
        if OUT.exists() else set()
    tmp = ROOT / "data" / "kit" / "critic_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    judged = 0
    for scene, r in scenes().items():
        if scene in done:
            continue
        bodies = list(r["prefix"]) + [extract_code(r["messages"][2]["content"])]
        intents = r["intents"]
        beats = [Beat(k + 1, None, t) for k, t in enumerate(intents)]
        # Exact beat ends: after each beat the scene records the renderer's
        # clock and holds half a second; the times are printed at the end,
        # and each frame is taken inside its beat's hold.
        held = [(b + "\nself.wait(0.5)\n_beat_ends.append(self.renderer.time)")
                if b.strip() else b for b in bodies]
        first = next(k for k, b in enumerate(held) if b.strip())
        held[first] = "_beat_ends = []\n" + held[first]
        last = max(k for k, b in enumerate(held) if b.strip())
        held[last] += "\nprint('BEAT_ENDS', _beat_ends)"
        asm = assemble(beats, held, kit=True)
        if not asm.ok:
            continue
        res = h.render(asm.code, quality="low", frames=2, use_cache=False)
        if not res.ok or not res.video_path:
            continue
        m = re.search(r"BEAT_ENDS \[([^\]]*)\]", res.stdout or "")
        ends = [float(x) for x in m.group(1).split(",")] if m and m.group(1) else []
        live = [k for k, b in enumerate(bodies) if b.strip()]
        if len(ends) != len(live):
            continue
        images, n = [], len(live)
        for k, t in zip(live, ends):
            f = tmp / f"{abs(hash(scene))}_{k}.jpg"
            subprocess.run(["/opt/homebrew/bin/ffmpeg", "-loglevel", "error", "-y",
                            "-ss", f"{max(t - 0.25, 0):.2f}", "-i", res.video_path,
                            "-frames:v", "1", "-vf", "scale=640:-1", str(f)])
            if f.exists():
                images.append(base64.b64encode(f.read_bytes()).decode())
                f.unlink()
        intents = [intents[k] for k in live]
        if len(images) != n:
            continue
        if dry:
            print(f"  {scene}: {n} beats, ends {[round(e, 1) for e in ends]}",
                  flush=True)
            return 0
        text = ASK.format(request=r.get("request", "")[:300],
                          beats="\n".join(f"{k + 1}: {t}" for k, t in
                                          enumerate(intents)))
        parts = [{"text": text}] + [{"inline_data": {"mime_type": "image/jpeg",
                                                     "data": im}}
                                    for im in images]
        body = json.dumps({"contents": [{"role": "user", "parts": parts}],
                           "generationConfig": {"temperature": 0,
                                                "maxOutputTokens": 400}}).encode()
        data = None
        for attempt in range(2 * len(MODELS)):
            model = MODELS[attempt % len(MODELS)]
            req = urllib.request.Request(f"{URL}/models/{model}:generateContent",
                                         data=body,
                                         headers={"x-goog-api-key": key,
                                                  "Content-Type": "application/json"})
            try:
                data = json.load(urllib.request.urlopen(req, timeout=120))
                break
            except urllib.error.HTTPError as e:
                print(f"  {scene}: HTTP {e.code} ({model})", flush=True)
                time.sleep(5 if attempt < len(MODELS) else 60)
            except (urllib.error.URLError, TimeoutError):
                time.sleep(10)
        if data is None:
            for _ in range(20):                 # the daily quota; wait it out
                time.sleep(60)
            continue
        reply = "".join(p.get("text", "") for p in
                        data["candidates"][0]["content"]["parts"])
        verdicts = {live[int(m.group(1)) - 1]: m.group(2).upper()
                    for m in re.finditer(r"^\s*(\d+)\s*[:.)-]\s*(YES|NO)", reply,
                                         re.M | re.I)
                    if 0 < int(m.group(1)) <= len(live)}
        with OUT.open("a") as f:
            f.write(json.dumps({"scene": scene, "verdicts": verdicts,
                                "raw": reply[:600]}) + "\n")
        judged += 1
        yes = sum(v == "YES" for v in verdicts.values())
        print(f"  {scene}: {yes}/{len(verdicts)} beats pass", flush=True)
    print(f"judged {judged} scenes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
