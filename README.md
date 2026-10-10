# Manim Forge

Type a sentence — *"a jacket costs £80 and is 15% off, what do you pay?"* —
and get a short narrated 3Blue1Brown-style animation, written, rendered and
checked by a local model on a Mac. No cloud, no API key.

Free, non-commercial, and built in the open — including the parts that went
wrong ([`docs/POSTMORTEM.md`](docs/POSTMORTEM.md),
[`docs/RESULTS.md`](docs/RESULTS.md)).

---

## How it works

```
request ──► retrieve ──► plan, then ──► render ──► critic ──► rewrite once ──► voice ──► video
            the 2 nearest   write the whole    with the     (what the kit   with the       (Kokoro,
            of 966 hand-    scene in one call  Forge kit    saw, idle beats, problems      local)
            written scenes  (3–6 beats)        (64 blocks)  slips, errors)  listed
```

- **One call writes the whole scene**, beat by beat in one context, so later
  beats reuse and transform what earlier beats built
  (`forge/app/oneshot.py`). It first writes a short lesson plan as comments
  (the idea, the picture, the numbers, the arc). The model is **Qwen3.5-9B,
  untuned**, 4-bit, running under MLX.
- **It adapts rather than invents.** The two nearest of 966 hand-written,
  checked scenes (TF-IDF + a small embedding, `forge/kit/library.py`) are in
  the prompt, along with the kit's reference.
- **The Forge kit** (`forge/kit/kit.py`): 64 blocks — `apply_matrix`,
  `slide_tangent`, `riemann_refine`, `bayes_square`, … — that draw and animate
  the picture so a small model's calls still produce a clean frame.
- **A critic, then one rewrite** (`forge/app/critique.py`): the kit reports
  in words what it saw (a point off its axes, a curve cut off, text on text,
  an empty frame), the critic adds beats that only change the caption,
  arithmetic slips (`forge/app/checks.py`) and render errors, and the draft
  goes back to the model with that list.
- **Narrated:** each beat's line is spoken by Kokoro-82M and the beat is
  held until it is said (`forge/app/voice.py`).

## Where it stands

Graded **by eye**, blind, from contact sheets: **good** = right answer and the
pictures show it; **partial** = right answer, weak pictures or one flaw;
**bad** = wrong or broken. Grades and reasons per scene are in `data/eye/`.

**v1.5 against v1.0** on 20 fresh requests used for nothing else (a computer
adding binary, the moon's phases, a probability tree, a unit circle, …):

| | good | partial | bad |
|---|---|---|---|
| **v1.5** | **10** | 7 | 3 |
| v1.0 | 7 | 11 | 2 |
| v1.5 + a LoRA trained on its own best scenes | 8 | 9 | 3 |

v1.0 on its original sets: in-scope (school maths and everyday quantities)
12 / 6 / 2, held-out classic topics 9 / 4 / 7.

The honest scope: about half of new requests come out right with pictures
that show it, most of the rest have the right answer with a weak or flawed
picture, and about one in seven is wrong, usually a confident wrong number.
Every model this project trained, including the v1.5 LoRA, did no better than
the untuned model with good context and checks (`docs/RESULTS.md`,
`docs/HISTORY.md`). The ten v1.0 scenes are in [`docs/GALLERY.md`](docs/GALLERY.md);
the v1.5 changes in [`docs/RELEASE_v1.5.md`](docs/RELEASE_v1.5.md).

## Run it

Apple Silicon Mac, 16 GB.

```bash
brew install cairo pango pkg-config ffmpeg
brew install --cask basictex          # MathTex needs LaTeX
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt mlx mlx-lm
./.venv/bin/pip install mlx-audio "misaki[en]"     # narration (optional)
export PATH="/Library/TeX/texbin:$PATH"
```

```bash
./.venv/bin/python -m forge.serve     # http://127.0.0.1:8765
```

The model (~6 GB) downloads on the first request and loads in ~20 s after
that. A narrated scene takes **~50 s**, or **~2.5 min** when the first draft
goes back for its rewrite. `FORGE_ADAPTER=hub` tries the v1.5 LoRA.

Tests and invariants:

```bash
./.venv/bin/python -m pytest
./.venv/bin/python -m forge.doctor
```

## Layout

```
forge/
  kit/          the Forge kit, the scene library, the hand-written teacher scenes
  app/          one shot (oneshot.py), checks, voice, and the older two-stage pipeline
  serve/        the local web app (FastAPI + server-sent events)
  harness/      render + error classification
  evaluate/     held-out prompts, local and Gemini vision judges
  synth/        teacher generation with provider rotation
  gold/         44 hand-authored scenes (the early style anchor)
scripts/        data building, critic, scorecard, judge, Kaggle launch, supervisor
kaggle/         the SFT and GRPO notebooks
```

## ManimCE, not ManimGL

Grant Sanderson's own code uses **ManimGL**; the docs, datasets and published
results use **ManimCE**. This builds on **ManimCE 0.21.0** and treats
`3b1b/videos` as style reference only.

## Documents

| | |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | the finish plan and how it got here |
| [`docs/RESULTS.md`](docs/RESULTS.md) | every measurement, with its confounds |
| [`docs/REASSESS.md`](docs/REASSESS.md) | the six-hour reassessments |
| [`docs/TEACHER_GUIDE.md`](docs/TEACHER_GUIDE.md) | how teacher scenes are written |
| [`docs/STATE.md`](docs/STATE.md) | resume with no prior context |
| [`docs/POSTMORTEM.md`](docs/POSTMORTEM.md) | the early defects, and what caused each |

## License

CC BY-NC-SA 4.0 — see [LICENSE](LICENSE). Non-commercial by design and by
obligation: the corpus inherits share-alike terms from its sources.
