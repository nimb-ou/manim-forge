"""Visual critic — the gate the render harness cannot be.

The render gate answers "did it run". It cannot answer "is it any good", and
those come apart constantly. The first version of the gold chess scene rendered
perfectly while drawing the percentage on top of the board: correct code,
unreadable output, and a clean pass.

Geometry checks catch some of that — a bounding box outside the frame, two
labels overlapping. But "does this actually communicate the idea" needs
something that can look. Gemini is multimodal and free at this tier, so it can
read a sampled frame and answer directly.

Two layers, cheapest first:

* ``geometric_report`` — pure pixel analysis, no API call. Catches content
  running off-frame and near-empty frames.
* ``critique`` — a multimodal judgement on sampled frames, returning structured
  scores so results are comparable across runs rather than prose.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

from forge.synth.teacher import GEMINI_REST, _load_dotenv

_load_dotenv()

RUBRIC = """You are a harsh reviewer checking a frame from a mathematical
animation for defects. Your job is to FIND PROBLEMS, not to be encouraging.

First look for these specific defects:
  - text or shapes clipped by the frame edge
  - anything drawn on top of anything else it obscures
  - text too small or too faint to read comfortably
  - elements crowded together with no breathing room
  - labels detached from, or ambiguous about, what they label
  - leftover elements from an earlier moment that no longer belong

Then score. Use the FULL range — most frames are not 5s:
  5 = a professional would publish this frame unchanged
  4 = minor imperfection, a careful reviewer would mention it
  3 = a viewer would notice something is off
  2 = a real defect that interferes with reading it
  1 = broken; the frame fails to communicate

If you find ANY defect from the list above, the relevant score is at most 3.
A frame with text sitting on top of a diagram is not a 4.

Reply with ONLY a JSON object, no prose:

{"readable": n,     // is all text legible and unobstructed?
 "in_frame": n,     // is anything clipped by the edges?
 "uncluttered": n,  // is there breathing room?
 "overlap": n,      // do elements collide or sit on top of each other?
 "communicates": n, // does the frame convey the stated intent?
 "worst_problem": "one short phrase naming the single worst defect, or empty string if genuinely none"}"""


@dataclass
class FrameVerdict:
    readable: int = 0
    in_frame: int = 0
    uncluttered: int = 0
    overlap: int = 0
    communicates: int = 0
    worst_problem: str = ""

    @property
    def mean(self) -> float:
        vals = [self.readable, self.in_frame, self.uncluttered,
                self.overlap, self.communicates]
        return round(sum(vals) / len(vals), 2)

    @property
    def is_broken(self) -> bool:
        """Any single dimension at 3 or below is a real defect, however good
        the rest. Averaging would let a badly clipped frame hide behind four
        good scores — and the rubric defines 3 as "a viewer would notice
        something is off", which is already too low to ship."""
        return min(self.readable, self.in_frame, self.overlap,
                   self.uncluttered, self.communicates) <= 3


def geometric_report(frame_path: str | Path) -> dict:
    """Free checks, no API call: is content touching the edges, or is the frame
    essentially blank?"""
    from PIL import Image
    import numpy as np

    img = Image.open(frame_path).convert("L")
    a = np.asarray(img, dtype=float) / 255.0
    h, w = a.shape

    # Manim renders on near-black; anything meaningfully brighter is content.
    content = a > 0.12
    coverage = float(content.mean())

    band = max(2, int(min(h, w) * 0.015))
    touching = bool(
        content[:band, :].any() or content[-band:, :].any()
        or content[:, :band].any() or content[:, -band:].any()
    )
    return {
        "coverage": round(coverage, 4),
        "touches_edge": touching,
        "near_blank": coverage < 0.004,
        "overcrowded": coverage > 0.42,
    }


def critique(frame_path: str | Path, intent: str,
             model: str = "gemini-3.5-flash-lite") -> FrameVerdict:
    """Ask a multimodal model to score one frame against its stated intent."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")

    raw = Path(frame_path).read_bytes()
    body = json.dumps({
        "contents": [{"role": "user", "parts": [
            {"text": f"{RUBRIC}\n\nThe frame is meant to show: {intent}"},
            {"inline_data": {"mime_type": "image/jpeg",
                             "data": base64.b64encode(raw).decode()}},
        ]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 800,
                             "responseMimeType": "application/json"},
    }).encode()

    req = urllib.request.Request(
        f"{GEMINI_REST}/models/{model}:generateContent",
        data=body, headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    data = json.load(urllib.request.urlopen(req, timeout=120))
    cands = data.get("candidates") or []
    if not cands:
        return FrameVerdict()
    txt = "".join(p.get("text", "")
                  for p in (cands[0].get("content", {}).get("parts") or []))
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        return FrameVerdict()
    return FrameVerdict(**{k: d.get(k, 0) for k in FrameVerdict().__dict__})
