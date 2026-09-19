# Manim Forge

A locally-trained model that turns plain English into 3Blue1Brown-style
mathematical animation — built on a corpus where **every example is proven to
render**.

Everything runs on-device: a MacBook Air M4 with 16 GB of unified memory. No API
calls, no cloud training, no inference costs.

**Build plan:** [Manim Forge plan](https://claude.ai/artifact/RUCpGv4xBv26Hkt34MiAHi)

---

## Why this is different

The public Manim datasets are text that was never executed. Scraped Manim code
spans six years of mutually incompatible API versions, so a model trained on it
learns a plausible-looking dialect that does not run.

Manim Forge render-gates everything. A row enters the corpus only after it has
been executed, rendered to video, and sampled to frames. It is a mechanical
quality filter that needs no human judgement per row, and it is the reason a 7B
model can be competitive here.

The second bet follows from published work on this exact task: **inference-time
render-and-retry beat the fine-tuning itself.** So the model is wrapped in a
loop that compiles its own output and reads its own tracebacks.

## Architecture

The render harness is one component doing three jobs — which is what keeps this
project from tripling in size:

| Phase | Role of the harness |
|-------|--------------------|
| 2 — render gate | the data filter |
| 6 — evaluation  | Render Success Rate *is* its pass rate |
| 7 — platform    | the self-repair loop inside the product |

```
forge/
  harness/    render + error classification   <- Phase 0  [working]
  ingest/     pull sources into one schema    <- Phase 1
  gate/       execute everything, keep what survives
  style/      the 3b1b style layer            <- Phase 3
  synth/      scale the corpus                <- Phase 4
  train/      MLX LoRA fine-tune              <- Phase 5
  evaluate/   RSR, SSIM, CLIP, style adherence
  app/        local FastAPI platform          <- Phase 7
```

## Status

- [x] **Phase 0** — Manim CE 0.21.0 verified rendering; harness built and
      passing acceptance tests across 7 failure modes
- [ ] Phase 0 — BasicTeX (`MathTex` currently fails; see Setup)
- [ ] Phase 1 — corpus ingestion
- [ ] Phase 2 — the render gate

## Setup

```bash
brew install cairo pango pkg-config ffmpeg
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

LaTeX is **required** — `MathTex` and `Tex` hard-fail without it, and a maths
animation engine that cannot typeset an equation is not the product:

```bash
brew install --cask basictex        # ~1.5 GB, asks for your password
```

Verify the harness:

```bash
./.venv/bin/python scripts/smoke_harness.py
```

Expected: `valid_no_latex` passes with frames extracted, and each broken case
reports its specific `error_kind`. Once BasicTeX is installed, `needs_latex`
must also pass.

## A note on Manim versions

Grant Sanderson's own code uses **ManimGL**. Everything else — the docs, the
datasets, every published result — uses **ManimCE**. They share a name and
little else.

Manim Forge builds on **ManimCE 0.21.0**, and treats `3b1b/videos` as *style
reference* rather than code to imitate. Mixing the two APIs in one training set
teaches a model to hallucinate across both. Corpus rows are tagged `ce` or `gl`
at ingest so they can never blend silently.

## License

CC BY-NC-SA 4.0 — see [LICENSE](LICENSE). Non-commercial by design and by
obligation; the corpus inherits share-alike terms from its sources.
