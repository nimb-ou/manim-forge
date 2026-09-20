# State — read this first

Everything needed to resume with no prior context. If this file and the repo
disagree, the repo is right and this file is stale — fix it.

## What this is

A locally-trained model that turns plain English into 3Blue1Brown-style Manim
animation. Runs on a MacBook Air M4, 16 GB. Never commercial; CC BY-NC-SA 4.0.
Repo: github.com/nimb-ou/manim-forge

## The one idea

**Every training row is proven to render.** Public Manim datasets are text that
was never executed, spanning years of incompatible API versions. The render
gate is a mechanical filter needing no per-row judgement, and it is the same
component that serves as the evaluation metric and the product's repair loop.

## Numbers that matter

| metric | value | where |
|---|---|---|
| Benchmark (single scenes) | **89%** | `docs/RESULTS.md` |
| Hard eval (whole explainer) | 85% render, 16s vs 16min, 8.8% coverage | same |
| Verified corpus | ~2,300 | `python -m forge.catalog` |
| Gold scenes | 10 of 61 | `python -m forge.gold.curriculum` |
| Free-tier budget | **~900–1,000 calls/day total** (not per model) | measured |

## Layout

```
forge/harness/     render + error classification — the load-bearing component
forge/ingest/      5 public sources -> one schema, AST-structural dedupe
forge/gate/        execute everything, keep what survives
forge/repair/      lint + API introspection + escalating repair loop
forge/retrieve/    few-shot retrieval over verified scenes (local embeddings)
forge/synth/       teacher generation, model rotation, task types
forge/primitives/  9 modules — the animation computes its own claims
forge/gold/        10 hand-authored scenes + the 61-scene curriculum
forge/evaluate/    benchmark, hard eval, visual critic
forge/app/         local FastAPI platform
forge/catalog.py   every dataset declared, with orphan detection
```

## Commands

```bash
python -m forge.catalog                     # what data exists, what is orphaned
python -m forge.gold.curriculum             # what to build next
./scripts/overnight.sh                      # supervisor: daemons + backup + index
python scripts/generate_forever.py          # code generation, rotates 6 models
python scripts/generate_tasks.py            # prose tasks (plan/narrate/critique/...)
python scripts/run_repair_benchmark.py --n 100 --retrieval   # the 89% number
python scripts/run_hard_eval.py --n 81 --backend local --retrieval
python scripts/backup_to_hf.py              # -> nimitttt/manim-forge-corpus (private)
uvicorn forge.app.server:app --port 8765    # the platform
```

`export PATH="/Library/TeX/texbin:$PATH"` before anything that renders.

## Hard-won facts

- **Rate limits are per model.** Rotation reaches the daily ceiling; it does
  not raise it. More workers made throughput *worse* (10/min → 5.7/min).
- **`Dot3D` costs 100× a flat `Dot`** — 19.3s vs 0.2s for 200 points.
- **Animations leave stage copies** the tracked reference no longer points at.
  `FadeOut(self.x)` is not enough; sweep by type or colour.
- **`set_opacity` on a VMobject dims fill too** — use `set_stroke(opacity=…)`.
- **`Rotate` preserves length.** Rotating by an angle difference gives an arrow
  of the wrong magnitude pointing the right way.
- **Transform between VGroups of different size** pairs members arbitrarily and
  smears. Fade instead.
- macOS **spawns** subprocesses — multiprocessing needs `if __name__ == …`.
- Gemini's new **`AQ.` keys** are rejected by the OpenAI-compat endpoint and by
  google-genai; raw REST accepts them.
- Truncated generations surface as **syntax errors**, not as truncation. Budget
  generously (16k) and check `finishReason`.

## Standing instructions from Nimit

- Preserve every artefact, failures included. Storage is not a constraint.
- Narrate the work in chat; teach as you go; show real output.
- High bar. Do not ship something that merely renders.
- One variable per experiment. (Broken three times so far; costs a week
  once training starts.)

## Next

1. Finish Tier 1 of the curriculum (visual grammar) — nothing should build on
   an unbuilt foundation.
2. Kaggle SFT once the corpus is large enough; must beat 89%.
3. GRPO with the render gate as reward — the week-long run.
4. Deploy: HF Space, Docker, API backend first, local model after.
