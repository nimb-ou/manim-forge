"""3Blue1Brown transcripts — narration paired with animation.

This is the one signal no public dataset has. Grant's videos carry his
narration, and the scene code for those same videos sits in ``3b1b/videos``,
organised per video. Together they give ``(what he says) -> (what appears)``
pairs from the person who defined the style.

Two uses, and only one of them is training data in the usual sense:

**Planner training.** Turning an explanation into a sequence of visual beats is
exactly the planner's job, and a transcript segmented by topic is a worked
example of it. This is the valuable half.

**Prompt realism.** Our synthetic prompts come from a topic list and read like
a topic list. Narration reads like a person explaining something, which is much
closer to what someone will actually type.

What this is *not*: a source of target code. His code is ManimGL, a different
library, and training a CE model on it teaches cross-API hallucination — the
failure the corpus tags ``flavor`` to prevent.

Licence: ``3b1b/videos`` is CC BY-NC-SA 4.0, which this project also carries.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

VTT_TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})")
TAGS = re.compile(r"<[^>]+>")


@dataclass
class Cue:
    start: float
    end: float
    text: str


def _secs(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_vtt(path: Path) -> list[Cue]:
    """Parse a WebVTT file into cues, de-duplicated.

    Auto-generated captions repeat each line as the next one rolls in, so raw
    cues contain the same sentence three or four times. Collapsing consecutive
    duplicates is what makes the text readable as prose.
    """
    cues: list[Cue] = []
    start = end = None
    buf: list[str] = []

    for raw in path.read_text(errors="replace").splitlines():
        m = VTT_TIME.search(raw)
        if m:
            if start is not None and buf:
                cues.append(Cue(start, end, " ".join(buf).strip()))
            start = _secs(*m.groups()[:4])
            end = _secs(*m.groups()[4:])
            buf = []
            continue
        line = TAGS.sub("", raw).strip()
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        buf.append(line)

    if start is not None and buf:
        cues.append(Cue(start, end, " ".join(buf).strip()))

    out: list[Cue] = []
    for c in cues:
        if out and (c.text == out[-1].text or c.text in out[-1].text):
            out[-1].end = c.end
            continue
        out.append(c)
    return out


def fetch(video_id: str, out_dir: Path, ytdlp: str = "./.venv/bin/yt-dlp") -> Path | None:
    """Download subtitles only — never the video. Manual subs preferred."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob(f"{video_id}*.vtt"))
    if existing:
        return existing[0]
    subprocess.run(
        [ytdlp, "--skip-download", "--write-sub", "--write-auto-sub",
         "--sub-lang", "en", "--sub-format", "vtt",
         "--output", str(out_dir / "%(id)s.%(ext)s"),
         f"https://www.youtube.com/watch?v={video_id}"],
        capture_output=True, timeout=300,
    )
    found = list(out_dir.glob(f"{video_id}*.vtt"))
    return found[0] if found else None


def segment(cues: list[Cue], target_seconds: float = 22.0) -> list[dict]:
    """Group cues into beat-sized spans of narration.

    Roughly 20 seconds because that is the span a handful of animation beats
    occupies — the unit the planner has to produce. Splits prefer sentence
    boundaries so a segment reads as a complete thought.
    """
    segs: list[dict] = []
    buf: list[Cue] = []

    for c in cues:
        buf.append(c)
        span = buf[-1].end - buf[0].start
        ends_sentence = buf[-1].text.rstrip().endswith((".", "?", "!"))
        if span >= target_seconds and ends_sentence:
            segs.append({"start": round(buf[0].start, 1), "end": round(buf[-1].end, 1),
                         "text": " ".join(c.text for c in buf)})
            buf = []

    if buf:
        segs.append({"start": round(buf[0].start, 1), "end": round(buf[-1].end, 1),
                     "text": " ".join(c.text for c in buf)})
    return segs
