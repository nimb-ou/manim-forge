"""Hard evaluation — a whole video from a single line.

ManimBench prompts describe one scene: "display a dot, a square and a triangle,
then animate them appearing one by one". Useful, and far easier than what this
tool is actually for. Nobody types that. They type a subject.

So this set takes real 3Blue1Brown video titles as one-line prompts and asks
for the whole explainer. It is deliberately much harder than the benchmark, and
early scores are expected to be poor — the point is a measurement that moves as
the model gets better at the job we actually care about, rather than one that
saturates on single scenes.

Scoring cannot be "does it match the real video" — his code is ManimGL and no
two explanations of a topic agree anyway. What can be measured honestly:

* **renders** — the floor, as always
* **duration** — does a one-line prompt yield seconds or minutes?
* **beats** — is it structured, or one undifferentiated block?
* **concept coverage** — do the generated narration lines mention the terms the
  real video spends its time on? Crude, but it distinguishes an animation
  *about* the subject from one that merely renders.
* **visual quality** — the critic, on sampled frames
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Words that carry no topic signal. Kept short on purpose: an aggressive stop
# list would strip the mathematical vocabulary that is the whole measurement.
_STOP = set("""
the a an and or but if then than that this these those of in on at to for with
from by as is are was were be been being it its it's he she they we you i not
no so such can could would should will shall may might must do does did done
what which who whom whose when where why how all any both each few more most
other some only own same too very just now also here there about into over
under again once because while during before after above below up down out off
one two three four five six seven eight nine ten first second next last thing
things way ways like really actually quite lot lots kind sort bit much many
going want need know think see look make made take get got give gives let lets
say says said mean means going im ive youre thats theres weve dont doesnt
have has had them their there's what's let's you're we're i'm don't isn't
your yours our ours mine hers his doing done being your something someone
anything everything nothing feel feels felt seem seems maybe perhaps
probably certainly obviously basically essentially simply merely already
still even ever never always often sometimes usually
""".split())

#: Titles that are not explainers. A Q&A or a podcast has no animation to
#: evaluate, and scoring one would measure nothing while dragging the average.
_NOT_EXPLAINER = re.compile(
    r"\bQ&A\b|\bpodcast\b|\bannounc|\blivestream\b|\binterview\b|"
    r"\bthank you\b|\bbehind the scenes\b|\bhow I animate\b|\bchannel\b",
    re.I)

_WORD = re.compile(r"[a-z][a-z'-]{2,}")


def key_terms(text: str, top: int = 22) -> list[str]:
    """The words a passage actually spends its time on."""
    words = [w for w in _WORD.findall(text.lower()) if w not in _STOP]
    return [w for w, _ in Counter(words).most_common(top)]


@dataclass
class HardTask:
    video_id: str
    title: str
    prompt: str
    real_seconds: float
    terms: list[str]


def build_tasks(narration: Path = Path("data/style/narration.jsonl"),
                min_segments: int = 20) -> list[HardTask]:
    """One task per video, with its real runtime and its vocabulary."""
    by_video: dict[str, list[dict]] = {}
    for line in narration.open():
        r = json.loads(line)
        by_video.setdefault(r["video_id"], []).append(r)

    tasks = []
    for vid, segs in by_video.items():
        if len(segs) < min_segments:
            continue
        segs.sort(key=lambda s: s["index"])
        title = segs[0]["title"]
        # Real runtime comes from the last cue's end time, not a words-per-minute
        # guess — the guess put a 20-minute video at 154 minutes.
        real_seconds = max(s["end"] for s in segs)
        full_text = " ".join(s["text"] for s in segs)

        # The prompt is the title, cleaned of series markers — a single line,
        # exactly what someone would type.
        prompt = re.sub(r"\s*\|.*$", "", title).strip()
        if _NOT_EXPLAINER.search(title):
            continue
        tasks.append(HardTask(vid, title, prompt, real_seconds,
                              key_terms(full_text)))
    tasks.sort(key=lambda t: t.real_seconds)
    return tasks


@dataclass
class HardScore:
    video_id: str
    title: str
    prompt: str
    ok: bool
    error_kind: str
    seconds: float = 0.0
    real_seconds: float = 0.0
    n_beats: int = 0
    n_play_calls: int = 0
    coverage: float = 0.0
    visual_mean: float = 0.0
    matched_terms: list[str] = field(default_factory=list)

    @property
    def length_ratio(self) -> float:
        return self.seconds / self.real_seconds if self.real_seconds else 0.0


def concept_coverage(code: str, terms: list[str]) -> tuple[float, list[str]]:
    """Fraction of the real video's key terms the generated scene mentions.

    Read from narration= strings and on-screen Text/MathTex, i.e. what a viewer
    would hear or see — not from variable names, which a model can pad without
    the animation saying anything.
    """
    spoken = " ".join(re.findall(r'narration\s*=\s*["\']([^"\']{10,})', code))
    shown = " ".join(re.findall(r'(?:Text|MathTex|Tex)\s*\(\s*r?["\']([^"\']+)', code))
    haystack = (spoken + " " + shown).lower()
    hit = [t for t in terms if t in haystack]
    return (len(hit) / len(terms) if terms else 0.0), hit
