# Manim Forge

Type a sentence — *"why the angles of a triangle add up to 180°"* — and get
a short 3Blue1Brown-style animation, planned, written, rendered and checked
by a local 7B model on a Mac.

Free, non-commercial, and built in the open — including the parts that went
wrong ([`docs/POSTMORTEM.md`](docs/POSTMORTEM.md)).

---

## How it works

```
request ──► planner ──► beats ──► kit coder ──► Forge kit ──► render ──► video
            (arc of 3–8      (one beat's        (64 animation       │
             beats, each      code at a time,    blocks that         └─ a failing
             a picture +      on top of the      draw and animate)      statement or
             narration)       earlier beats)                            beat is dropped,
                                                                        the rest re-rendered
```

- **Two fine-tuned adapters on Qwen2.5-Coder-7B** (QLoRA, trained on Kaggle,
  run locally with MLX): a *planner* that writes the arc and a *kit coder*
  that writes each beat.
- **The Forge kit** (`forge/kit/kit.py`): 64 blocks — `apply_matrix`,
  `slide_tangent`, `riemann_refine`, `bayes_square`, `wind_signal`, … — that
  take the calls a small model actually makes and still produce a clean frame.
  Fuzz-tested on 70 case groups (`scripts/fuzz_kit.py`).
- **Render-verified training data.** Every training row rendered; beats are
  kept only if a vision model, looking at the frame at the end of the beat,
  says it shows the beat's idea (`scripts/critic_kit_scenes.py`).

## Where it stands

The headline metric: for 20 held-out prompts on topics nothing was built
for, the share of planned beats whose end frame a vision judge (Gemini)
says shows a picture of the beat's idea (`scripts/judge_sheets.py`).
The 20 short prompts are the classic topics the kit and its teacher scenes
were built around, so they measure the familiar case. The held-out topics
were meant never to appear in training data; an audit on Oct 1 found that
some did (3Blue1Brown narration arcs on the central limit theorem and the
chain rule, a few gold and teacher scenes), so the numbers below are
somewhat optimistic. From kit v8 and planner v5 on, every builder drops a
row whose request is a held-out topic (`forge/evaluate/heldout_guard.py`).

| configuration | 20 short prompts | 20 held-out |
|---|---|---|
| raw Manim, fine-tuned | 24% | 25% |
| planner v3 + kit coder v6 (the app today) | 45–51% | 37–40% |

By eye, of 20 held-out scenes about 3 are clearly good. The current plan —
better plans, critic-filtered data, the next SFT round — is in
[`docs/PLAN.md`](docs/PLAN.md); every measurement, with its confounds, in
[`docs/RESULTS.md`](docs/RESULTS.md).

## Run it

Apple Silicon Mac, 16 GB.

```bash
brew install cairo pango pkg-config ffmpeg
brew install --cask basictex          # MathTex needs LaTeX
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt mlx mlx-lm
export PATH="/Library/TeX/texbin:$PATH"
```

The app expects the adapters in `adapters/mlx-planner3` and
`adapters/mlx-coder6-kit` (the base model downloads on first run):

```bash
./.venv/bin/python -m forge.serve     # http://127.0.0.1:8765
```

Tests and invariants:

```bash
./.venv/bin/python -m pytest
./.venv/bin/python -m forge.doctor
```

## Layout

```
forge/
  kit/          the Forge kit, block families, Claude-written teacher scenes
  app/          the two-stage pipeline: plan, write each beat, assemble, salvage
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
