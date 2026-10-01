"""Picture-first planner arcs from Gemma 4 on the Gemini API.

Planner v3's held-out plans are mostly beats no coder can draw ("Title
screen", "Introduce the concept of rational numbers", "Host explains the
rules"). The planner learns its intents from its training arcs, and most of
those came from narration transcripts. Gemma 4 31B has a free quota far
beyond the Flash models', so it writes arcs in bulk:

  1. topics: short requests per kit family, a person's phrasing, never the
     held-out topics -> data/kit/gemma_topics.jsonl
  2. arcs: 3-6 beats per topic, each intent naming a picture the kit can
     draw (the vocabulary below), with Claude's arcs as examples
     -> data/kit/gemma_plans.jsonl, in claude_plans.jsonl's row format
     (source "plan-gemma")

The arcs are not training data by themselves: self_kit_beats.py --plans
draws them with the kit coder, the critic judges each beat's frame, and only
arcs whose beats the coder could draw become planner rows
(build_coder_dataset.py --which planner).

    ./.venv/bin/python -u scripts/gemma_arcs.py --topics 40 --arcs 1000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forge.app.pipeline import PLAN_SYSTEM  # noqa: E402
from forge.evaluate.heldout_guard import HELDOUT  # noqa: E402
from forge.kit.families import FAMILIES  # noqa: E402

URL = "https://generativelanguage.googleapis.com/v1beta"
MODELS = ["gemma-4-26b-a4b-it", "gemma-4-31b-it"]   # 31b gave 500s on 2026-09-29
TOPICS = ROOT / "data" / "kit" / "gemma_topics.jsonl"
OUT = ROOT / "data" / "kit" / "gemma_plans.jsonl"
TEACHER = ROOT / "forge" / "kit" / "teacher"

HELD_OUT = ("harmonic series; chain rule; matrix multiplication as composition; "
            "Monty Hall; compound interest and e; logarithms; birthday paradox; "
            "standard deviation; least squares; merge sort; Pascal's triangle; "
            "roots of unity; mean value theorem; exponential decay or half-life; "
            "adding vectors tip to tail; square root of 2 irrational; Fibonacci or "
            "golden ratio; Fourier transform of a chord; determinant zero; "
            "central limit theorem")
# One pattern for every builder (it catches all 20 held-out prompts).
BANNED = HELDOUT

PICTURES = """What the animation can draw (describe pictures in these terms, never code):
- a coordinate grid with arrows (vectors), a matrix transforming the whole grid,
  the unit square's area changing, eigen-directions staying put, a vector's
  projection onto a line, a second basis's grid
- axes with a graph of a function, a tangent line sliding along it with its
  slope read out, a secant shrinking to a tangent, shaded area under a curve,
  rectangles under a curve getting thinner, a dot tracing the curve, a ball
  stepping downhill on a curve, polynomials closing in on a curve, a band
  narrowing around a limit, a temperature curve smoothing out
- a number line with marked points; bars of given heights; bars of partial
  sums growing; squares filling halves, quarters, eighths of a square
- a circle cut into slices and unrolled into a near-rectangle; a right
  triangle with squares on its sides; a triangle's angles laid on a line;
  polygons, circles, dots, arrows and lines at given coordinates
- a 6 by 6 grid of two-dice outcomes with a sum highlighted; coins flipping
  with the share of heads settling; a histogram growing from random samples;
  a probability square split by a prior and a test (Bayes)
- a rotating radius drawing a sine wave; a travelling wave; waves adding;
  a square wave built from sines; a signal wound around a circle (Fourier);
  a swinging pendulum with its angle traced over time
- the complex plane, multiplying by a complex number as rotate-and-scale,
  e^{it} walking the unit circle
- a vector field with arrows, particles flowing along it
- a neural network of layers of dots; a network graph of labelled nodes and
  edges; an array of bars being swapped (sorting); a convolution sliding one
  list over another; binary digits counting; a grid of bits; Towers of Hanoi;
  primes in a spiral; a sieve crossing out multiples
- equations (a derivation step by step) beside the picture, a title, a short
  caption, labels on objects"""

ASK_TOPICS = """List {n} different short requests a curious person might type to get a
short animated maths or science explanation about {family}. Each is 3 to 12
words, plain English, one idea, like "why a derivative is a slope" or "what
an eigenvector is". Every one must be something that can be explained with
pictures like these:

{pictures}

Do NOT include any of these topics: {held}.
Answer as a JSON list of strings, nothing else."""

ASK_ARC = """You are planning a short 3Blue1Brown-style animation for the request:
"{request}"

{pictures}

Write the plan as 3 to 6 beats (20-60 seconds in all). Every beat's intent is
ONE line that names the PICTURE on screen -- what is drawn and what moves --
with concrete numbers, functions or coordinates. Not "Introduce X" or
"Explain Y" or "Title screen": a viewer must be able to see the idea. Each
beat builds on the last; the whole arc explains the request correctly.
The narration is one or two spoken sentences.

Examples of good plans:
{examples}

Answer as JSON: {{"beats": [{{"seconds": 10, "intent": "...", "narration": "..."}}, ...]}}"""

BAD_INTENT = re.compile(r"^\s*(title|introduc|explain|welcome|text|summary|recap|"
                        r"conclusion|outro|intro\b|overview|definition of)", re.I)


def gemini_key() -> str:
    from forge.synth.teacher import Teacher
    return Teacher(provider="gemini", model="gemini-flash-latest")._key


def ask(key: str, text: str, max_tokens: int = 8000) -> str | None:
    body = json.dumps({"contents": [{"role": "user", "parts": [{"text": text}]}],
                       "generationConfig": {"temperature": 0.8,
                                            "maxOutputTokens": max_tokens}}).encode()
    for attempt in range(8):
        model = MODELS[attempt % len(MODELS)]
        req = urllib.request.Request(f"{URL}/models/{model}:generateContent", data=body,
                                     headers={"x-goog-api-key": key,
                                              "Content-Type": "application/json"})
        try:
            data = json.load(urllib.request.urlopen(req, timeout=180))
            parts = data["candidates"][0]["content"]["parts"]
            # Gemma may return its thinking as a part flagged "thought"
            return "".join(p.get("text", "") for p in parts if not p.get("thought"))
        except urllib.error.HTTPError as e:
            print(f"    HTTP {e.code} ({model})", flush=True)
            time.sleep(15 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError,
                ConnectionError, OSError):          # RemoteDisconnected, resets
            time.sleep(10)
    return None


class GemmaTeacher:
    """Teacher.ask's shape on Gemma's native API, its thinking dropped (the
    OpenAI-compatible path returns the thinking as the answer)."""

    def __init__(self):
        self.key = gemini_key()

    def ask(self, prompt: str, max_tokens: int = 8000, system: str = "") -> str:
        got = ask(self.key, (system + "\n\n" if system else "") + prompt,
                  max(max_tokens, 8000))
        if got is None:
            raise RuntimeError("gemma: no answer (429/500)")
        return got


def json_in(text: str):
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    s = m.group(1) if m else text
    start = min([i for i in (s.find("["), s.find("{")) if i >= 0], default=-1)
    if start < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(s[start:])[0]
    except json.JSONDecodeError:
        return None


def examples(k: int = 3) -> str:
    arcs = []
    for f in sorted(TEACHER.glob("plans_*.json")):
        arcs += json.loads(f.read_text())
    out = []
    for a in random.sample(arcs, k):
        lines = "\n".join(f"  {j + 1}. [{b.get('seconds', 10)}s] {b['intent']} -- "
                          f"{b.get('narration', '')}" for j, b in enumerate(a["beats"]))
        out.append(f'Request: "{a["request"]}"\n{lines}')
    return "\n\n".join(out)


def make_topics(key: str, per_family: int) -> list[str]:
    have = [json.loads(l)["request"] for l in TOPICS.open()] if TOPICS.exists() else []
    seen = {t.lower() for t in have}
    for fam in FAMILIES:
        got = json_in(ask(key, ASK_TOPICS.format(n=per_family, family=fam,
                                                 pictures=PICTURES, held=HELD_OUT)) or "")
        new = [t.strip() for t in (got or []) if isinstance(t, str)
               and 2 < len(t.split()) <= 14 and not BANNED.search(t)
               and t.strip().lower() not in seen]
        with TOPICS.open("a") as f:
            for t in new:
                seen.add(t.lower())
                f.write(json.dumps({"request": t, "family": fam}) + "\n")
        have += new
        print(f"  topics {fam}: +{len(new)} ({len(have)})", flush=True)
    return have


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=int, default=40, help="new topics per family (0: none)")
    ap.add_argument("--arcs", type=int, default=1000)
    a = ap.parse_args()
    key = gemini_key()
    topics = make_topics(key, a.topics) if a.topics else \
        [json.loads(l)["request"] for l in TOPICS.open()]
    done = {json.loads(l)["meta"]["request"] for l in OUT.open()} if OUT.exists() else set()
    todo = [t for t in topics if t not in done][: a.arcs]
    print(f"{len(done)} arcs done, {len(todo)} to go", flush=True)
    for n, req in enumerate(todo):
        got = json_in(ask(key, ASK_ARC.format(request=req, pictures=PICTURES,
                                              examples=examples())) or "")
        bs = (got or {}).get("beats") if isinstance(got, dict) else None
        if not bs or not 3 <= len(bs) <= 6 or not all(
                isinstance(b, dict) and b.get("intent") for b in bs):
            print(f"  skip {req[:50]}", flush=True)
            continue
        if any(BAD_INTENT.search(b["intent"]) or len(b["intent"]) > 160 for b in bs):
            print(f"  reject (intent) {req[:50]}", flush=True)
            continue
        text = "\n".join(f"{j + 1}. [{int(b.get('seconds') or 10)}s] {b['intent'].strip()} -- "
                         f"{str(b.get('narration', '')).strip()}".rstrip(" -")
                         for j, b in enumerate(bs))
        row = {"messages": [
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": f"REQUEST\n{req}\n\nBEATS SO FAR\n"
             "  (nothing yet — open the explanation)\n\nWrite the next 6 beat(s), "
             "numbered from 1. Stop early and write END if the explanation is complete."},
            {"role": "assistant", "content": text + "\nEND"}],
            "meta": {"id": f"plan-gemma:{len(done) + n}", "source": "plan-gemma",
                     "task": "plan-window", "n_beats": str(len(bs)), "request": req,
                     "final": "True"}}
        with OUT.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  ok   {req[:60]} ({len(bs)} beats)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
