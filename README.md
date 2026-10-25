# Manim Forge

Type a sentence — *"a jacket costs £80 and is 15% off, what do you pay?"* —
and get a short narrated animation in the style of 3Blue1Brown, written,
rendered and checked by a local model on a Mac. No cloud, no API key.

**Students: start with [`docs/STUDENTS.md`](docs/STUDENTS.md)** — install
with one command, then double-click *Start Manim Forge*.

Free, non-commercial, and built in the open — including the parts that went
wrong ([`docs/POSTMORTEM.md`](docs/POSTMORTEM.md),
[`docs/RESULTS.md`](docs/RESULTS.md)).

---

## How it works

```
request ──► work out ──► retrieve ──► plan, then ──► render ──► critic ──► rewrite ──► voice ──► video
            the numbers   the 2 nearest  write the whole  with the    (wrong or    once        (Kokoro,   + "checked"
            (a Python     of 966 hand-   scene in one     Forge kit   unexplained              local)       or "check
            program, run) written scenes call (3–6 beats)             numbers, …)                           this" badge
```

- **Compute, don't guess** (`forge/app/solve.py`). Before the scene is
  written, the model writes a short Python program that works out every
  number the explanation needs, checking its answer a second way with an
  `assert`. It is run; its variables are handed to the scene and defined in
  it, so numbers on screen come from the calculation. Any number shown that
  the calculation did not produce, a worked answer missing from the screen,
  or a point drawn off its curve is reported — to the model for its
  rewrite, and to the student as a **"check this one"** badge instead of
  **"numbers checked"**.

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

Apple Silicon Mac, 16 GB, ~12 GB free disk.

```bash
bash install.sh
```

It installs Homebrew if needed, the drawing tools (cairo, pango, ffmpeg,
BasicTeX), Python 3.12 and a private `.venv` with the pinned app
requirements (`requirements-app.txt`), downloads the models (~6 GB) and runs
a self-check (`python -m forge.serve.setup`). Then double-click
**Start Manim Forge** (or `./.venv/bin/python -m forge.serve`, port 8765).

The model loads in ~20 s on the first question. A narrated scene takes one
to three minutes. `FORGE_ADAPTER=hub` tries the v1.5 LoRA. The research
toolkit (training, judges, Kaggle) needs `requirements.txt` as well.

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
