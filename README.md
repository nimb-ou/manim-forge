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
request ──► retrieve ──► one call writes ──► checks ──► Forge kit ──► render ──► voice ──► video
            the 2 nearest   the whole scene     (arithmetic   (64 animation   (failing     (Kokoro,
            of 942 hand-    (3–6 beats, code    on screen,    blocks)          statements   local)
            written scenes  + narration)        layout, renders)               dropped)
```

- **One call writes the whole scene**, beat by beat in one context, so later
  beats reuse and transform what earlier beats built
  (`forge/app/oneshot.py`). The model is **Qwen3.5-9B, untuned**, 4-bit,
  running under MLX.
- **It adapts rather than invents.** The two nearest of 942 hand-written,
  checked scenes (TF-IDF + a small embedding, `forge/kit/library.py`) are in
  the prompt, along with the kit's reference.
- **The Forge kit** (`forge/kit/kit.py`): 64 blocks — `apply_matrix`,
  `slide_tangent`, `riemann_refine`, `bayes_square`, … — that draw and animate
  the picture so a small model's calls still produce a clean frame.
- **Checks before you see it:** on-screen arithmetic is evaluated
  (`forge/app/checks.py`), text collisions and off-frame objects are counted
  at each beat (`kit.layout_issues`), and a scene that fails any of them, or
  fails to render whole, is sampled once more.
- **Narrated:** each beat's line is spoken by Kokoro-82M and the beat is
  held until it is said (`forge/app/voice.py`).

## Where it stands

Graded **by eye** from contact sheets (the vision judges proved lenient),
20 prompts per set: **good** = right answer and the pictures show it;
**partial** = right answer on screen, weak pictures or one flaw; **bad** =
wrong or broken. Grades and reasons per scene are in `data/eye/`.

| | in-scope: good / partial / bad | held-out: good / partial / bad |
|---|---|---|
| **v1.0 as shipped (Mac, MLX, best of 2)** | **12 / 6 / 2** | **9 / 4 / 7** |
| same model, one sample, on Kaggle | 13 / 5 / 2 | 8 / 8 / 4 |
| one shot, Qwen2.5-Coder-7B, untuned | 10 / 8 / 2 | — |
| one shot, Qwen2.5-Coder-7B, fine-tuned for it | 6 / 10 / 4 | 3 / 4 / 13 |
| planner + per-beat coder, fine-tuned (the old app) | 0 / 13 / 7 | ~3 good |

*In-scope*: new numbers and contexts for question types the hand-written
scenes cover — school maths and everyday quantities
(`forge/evaluate/inscope_prompts.json`). *Held-out*: 20 classic topics the
library deliberately has nothing on (`forge/evaluate/heldout_prompts.json`),
graded with those scenes removed; the shipped app keeps them in.

The honest scope: questions like the ones in the in-scope set come out
right about 60% of the time and with the right answer on screen about 90%;
famous university topics come out right about half the time; when it is
wrong it is usually a confident wrong number. See the ten scenes in
[`docs/GALLERY.md`](docs/GALLERY.md). The fine-tuned models
this project trained are not in v1.0 — retrieval, a stronger base model and
checks beat every one of them (`docs/RESULTS.md`, Oct 7–8).

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
that. Tested Oct 8 from these steps in the browser: a narrated scene in
**~45–50 s**, or **~3 min** when the first sample fails a check and a second
is drawn.

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
