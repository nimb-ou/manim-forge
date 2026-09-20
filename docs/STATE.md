# State — read this first

Everything needed to resume with no prior context. If this file and the repo
disagree, the repo is right and this file is stale — fix it.

*Every figure here is produced by `python -m forge.doctor`. If this file and
the repo disagree, the repo is right — and the doctor will say so.*

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

Re-derived, never copied forward. `python -m forge.doctor` regenerates
the block below and exits non-zero if it has drifted; `--write` updates it.

<!-- doctor:begin -->

| metric | value |
|---|---|
| Scraped corpus | 3,680 deduplicated rows |
| — verified by the gate | **1,798** (48.9%) |
| — of those, rescued by lint | 38 (no API cost) |
| Synthetic, verified | **667** unique |
| Gold scenes, authored | **44 of 61** · 182 beats |
| Gold rendered at 1080p60 | **19 of 44** |
| Training mix | **3,010** train / 131 valid |
| 3b1b narration segments | 5,825 |

*Re-derived by `python -m forge.doctor`. Do not edit by hand.*
<!-- doctor:end -->

| | |
|---|---|
| Benchmark, single scenes | **93%** at repair rounds=4 |
| Hard eval, whole explainer | 85% render · 16.3s vs 16 min · 8.8% coverage |
| Free-tier budget | ~900–1,000 calls/day *total*, not per model |

**Nothing has been trained yet.** Every number above is the untuned
Qwen2.5-Coder-7B with inference-time scaffolding. That is the single largest
outstanding item.

## Layout

```
forge/harness/     render + error classification — the load-bearing component
forge/ingest/      5 public sources -> one schema, AST-structural dedupe
forge/gate/        execute everything, keep what survives
forge/repair/      lint + API introspection + escalating repair loop
forge/retrieve/    few-shot retrieval over verified scenes (local embeddings)
forge/synth/       teacher generation, model rotation, task types
forge/primitives/  11 modules — the animation computes its own claims
forge/gold/        44 hand-authored scenes + the 61-scene curriculum
forge/evaluate/    benchmark, hard eval, visual critic
forge/orchestrator.py  job model: health is progress, never existence
forge/app/         local FastAPI platform
forge/catalog.py   every dataset declared, with orphan detection
```

## Commands

```bash
python -m forge.catalog                     # what data exists, what is orphaned
python -m forge.gold.curriculum             # what to build next
./scripts/forge_start.sh                    # the always-on pool
./scripts/forge_stop.sh                     # and stopping it, properly
python scripts/forge_run.py --status        # one report, starts nothing
python scripts/register_gold.py --scene ... --prompt ...   # finish a gold scene
python scripts/prepare_training.py          # rebuild the training mix
python scripts/run_repair_benchmark.py --n 100 --retrieval   # the 93% number
python scripts/run_hard_eval.py --n 81 --backend local --retrieval
python scripts/backup_to_hf.py              # -> nimitttt/manim-forge-corpus (private)
uvicorn forge.app.server:app --port 8765    # the platform
```

`export PATH="/Library/TeX/texbin:$PATH"` before anything that renders.

## Hard-won facts

- **Rate limits are per model.** Rotation reaches the daily ceiling; it does
  not raise it. More workers made throughput *worse* (10/min → 5.7/min).
- **`Dot3D` costs 100× a flat `Dot`** — 19.3s vs 0.2s for 200 points.
  Use `resolution=(3,3)` for point clouds: 10× faster to build, pixel-identical
  at `radius=0.03`.
- **Manim finds scenes by `__name__`, not by module attribute.** A factory
  returning `type("P", ...)` for every variant produces four classes called
  `P`; asking for `R2` makes manim print a numbered menu and **block on stdin
  forever** at 0% CPU. Always `</dev/null` a batch render.
- **`subprocess.run(timeout=...)` kills one pid, not the process group.**
  Manim spawns ffmpeg and LaTeX; a timeout orphans them. `start_new_session`
  only helps if something signals the group afterwards. See `_killpg`.
- **Escalating to SIGKILL "only if the parent is still alive" never fires.**
  The parent dies promptly; its pool does not. Signal the group both times.
- **Animations leave stage copies** the tracked reference no longer points at.
  `FadeOut(self.x)` is not enough; sweep by type or colour.
- **`FadeOut` does not remove a mobject from the scene** — it is still
  rasterised every frame. `self.remove()` after fading, or a 600-point cloud
  costs 45 minutes.
- **`%` is a comment character in LaTeX.** `f"{x:.1%}"` inside `MathTex`
  fails, and reports itself as *"installation does not support converting PDF
  to SVG"*. Lint rule: `escape_percent`.
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

## Training on Kaggle — nine runs' worth of facts

Every one of these cost a run. None is in Kaggle's documentation where I
looked.

- **Datasets mount at `/kaggle/input/datasets/<owner>/<slug>/`**, not
  `/kaggle/input/<slug>`. Three runs died on that constant, each reporting
  the directory "does not exist at all" -- true of the path, false of the
  data. `find_data()` searches for a file it needs instead of assuming.
- **Kaggle unpacks archives on upload.** A shipped `forge.tar.gz` arrives as
  `forge/forge/` and the archive is gone.
- **`kernel_type: "script"` is plain Python.** `!pip` is a SyntaxError there.
  Check with `ast.parse` on the file *unmodified* -- stripping magics first
  to make the check pass is how this shipped.
- **A new dataset version is not immediately mountable**, and neither
  `dataset_status` nor the file listing nor the version number tells you
  when it is -- all three describe the previous version for a while. Upload
  a fingerprint file and wait until downloading it returns the new hash.
- **Status strings are `KernelWorkerStatus.ERROR` in capitals**, inside a
  sentence that also contains the kernel slug. Match the extracted token,
  not a substring of the message.
- **The kernel log is a JSON array** of `{stream_name, data}`, so `tail` on
  it shows one enormous line. `scripts/read_kaggle_log.py`.
- **Kaggle notebooks see none of the runner's environment.** `HF_TOKEN` has
  to be attached as a Kaggle Secret through the web UI, so anything needing
  it belongs in the workflow, not the kernel.

### QLoRA on a T4, specifically

- **trl 1.13 renamed `max_seq_length` to `max_length` and dropped
  `warmup_ratio`** (only `warmup_steps` remains). Pin the version: an
  unpinned `trl>=0.12` means a different training config every run.
- **`loss_type` defaults to `chunked_nll`**, which patches the LM head and
  assumes `forward` is a bound method. On a bitsandbytes-quantised model it
  is a `functools.partial`, and `SFTTrainer` cannot be constructed at all.
  Use `loss_type="nll"` -- at the cost of ~620 MB of un-chunked logits at
  seq=2048 over a 152k vocabulary.
- **`prepare_model_for_kbit_training` is not optional.** Without it,
  Qwen2.5's bf16 config leaves bf16 gradients, `fp16=True` turns on a
  GradScaler, and torch has no bf16 CUDA kernel for the AMP unscale:
  `NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda`.
  It upcasts fp16/bf16 params to fp32 -- which costs ~2.2 GB on Qwen's
  embedding table, so print the memory budget.
- **The GPU is 2x Tesla T4, 15360 MiB each**, and only one is used here.

## Standing instructions from Nimit

- Preserve every artefact, failures included. Storage is not a constraint.
- Narrate the work in chat; teach as you go; show real output.
- High bar. Do not ship something that merely renders.
- One variable per experiment. (Broken three times so far; costs a week
  once training starts.)

## Where the work stands

Done and verified:

- The pipeline end to end: ingest → gate → repair → retrieve → evaluate.
- 44 gold scenes, all importing cleanly, 182 beats, narration audit clean.
- A training mix of 2,989 examples with the composition written into the file.
- Serving architecture costed against measured latency (`docs/ARCHITECTURE.md`).

Not done:

1. **Train something.** No fine-tune has been run. Every number is untuned.
2. 17 gold scenes remain (3 Tier 2, 14 Tier 3) — `python -m forge.gold.curriculum`.
3. 25 gold scenes not yet rendered at 1080p60.
4. GRPO with the render gate as reward — the week-long run.
5. Deploy. Architecture written, nothing built.
6. Public dataset release pending a licensing review.

See `docs/RESULTS.md` for what has been measured and `docs/POSTMORTEM.md` for
what went wrong and what was changed so it cannot recur.
