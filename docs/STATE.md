# State — read this first

Everything needed to resume with no prior context. If this file and the repo
disagree, the repo is right and this file is stale — fix it. The full story,
mistakes included, is `docs/HISTORY.md`; every measurement is in
`docs/RESULTS.md`; the remaining phases are at the top of `docs/PLAN.md`.

## What this is (v1.5, 2026-10-10)

A request in plain English becomes a short, narrated 3Blue1Brown-style
animation, made entirely on a 16 GB Apple Silicon Mac. Never commercial;
CC BY-NC-SA 4.0. Repo: github.com/nimb-ou/manim-forge

## How v1.5 works

1. **Retrieve** the two nearest of 966 hand-written, checked kit scenes
   (`forge/kit/library.py`: TF-IDF + bge-small).
2. **One call writes the whole scene**: a `# plan:` lesson plan, then beats,
   narration and code, with the kit's reference in the prompt
   (`forge/app/oneshot.py`, `plan=True`). Model: **Qwen3.5-9B, untuned**, MLX
   4-bit (`oneshot.BASE_MODEL`), thinking off.
3. **Render** with the Forge kit (64 blocks, `forge/kit/kit.py`), salvaging a
   failing statement before a whole beat (`forge/app/pipeline.finish`). The
   kit reports what it repaired or saw as `ISSUE` lines (`Stage.issue`,
   `layout_report`).
4. **Critique and rewrite once** (`forge/app/critique.py`, `revise=1`): kit
   issues, caption-only beats, arithmetic slips and a failed render's error go
   back to the model with the draft; the better of the two by `oneshot.score`
   is kept.
5. **Narrate** each beat with Kokoro-82M and hold the beat until its line is
   said (`forge/app/voice.py`).

Served by `python -m forge.serve` (http://127.0.0.1:8766 via
`.claude/launch.json`, 8765 by default). ~50 s a scene; ~2.5 min with the
rewrite. Off by default: the v1.5 LoRA (`FORGE_ADAPTER=hub`), the self-check
(`run_oneshot(check=True)`).

## How good it is (by eye; `data/eye/`)

Fresh test set (20 requests, graded blind three ways, Oct 10):

| | good | partial | bad |
|---|---|---|---|
| **v1.5** | **10** | 7 | 3 |
| v1.0 | 7 | 11 | 2 |
| v1.5 + LoRA | 8 | 9 | 3 |

v1.0 on its own sets (Oct 8):

| | good | partial | bad |
|---|---|---|---|
| in-scope (school maths, everyday quantities) | 12 | 6 | 2 |
| held-out (20 classic topics, library scenes removed) | 9 | 4 | 7 |

On the Mac's own MLX engine, best of 2 (`is_v1`, `held_v1`). Gallery:
`docs/GALLERY.md`.

## What did not work, so do not repeat it blindly

Fine-tuning for this task (seven kit/coder/planner adapters, GRPO, a
retrieval-augmented SFT): each taught format and lost reasoning; the best
fine-tune was beaten by the untuned base with good context. Adding hand-
written scenes as *training data* plateaued (+1 point for ~400 scenes);
adding them to the *library* helps at once. The local vision judge is too
lenient to decide anything. Details: `docs/HISTORY.md`.

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
| Gold rendered at 1080p60 | **44 of 44** |
| Training mix | **3,010** train / 131 valid |
| 3b1b narration segments | 5,825 |

*Re-derived by `python -m forge.doctor`. Do not edit by hand.*
<!-- doctor:end -->

## Layout

```
forge/kit/        the Forge kit, the scene library, teacher scenes (forge/kit/teacher)
forge/app/        oneshot.py, critique.py (v1.5), checks.py, voice.py, pipeline.py (assembly,
                  salvage, render), twostage.py (the older planner + coder)
forge/serve/      the web app (FastAPI + server-sent events)
forge/harness/    render + error classification
forge/evaluate/   prompt sets (world = dev, fresh = test, inscope, heldout), held-out guard
scripts/          scorecard.py, judge_sheets.py, judge_scenes.py, make_gallery.py,
                  stack_sheets.py and blind_pairs.py (eye grading), selfgen.py +
                  build_selftrain.py (self-training rows), kaggle eval queues
kaggle/           SFT kernels, selftrain (Qwen3.5-9B QLoRA on a T4), oneshot_eval
data/eye/         by-eye grades with a reason per scene
docs/gallery/     the v1.0 gallery
```

## Commands

```bash
export PATH="/Library/TeX/texbin:$PATH"            # before anything that renders
./.venv/bin/python -m forge.serve                   # the app
./.venv/bin/python -m pytest -q                     # gates every commit
./.venv/bin/python -m forge.doctor                  # invariants (CI runs it too)
./.venv/bin/python -u scripts/scorecard.py --fresh --oneshot --api --coder none --samples 2 --revise 1 --plan --tag T
./.venv/bin/python scripts/stack_sheets.py T 01 02 03 04   # sheets to grade by eye
./.venv/bin/python scripts/blind_pairs.py T1 T2 [T3]        # blind comparison pages
./.venv/bin/python -u scripts/make_gallery.py       # gallery candidates
```

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

- Work autonomously; condensed updates; judge by evidence (sheets by eye).
- Commit only after pytest passes; check CI with `gh run list -L 2`.
- Never train on the Mac; one 7B/9B model in memory at a time.
- Held-out topics never in training data or the evaluation library.
- Long jobs: caffeinate, launched from bash (zsh niced `&` jobs), on AC
  power (`pmset -g batt`). Kaggle calls under a deadline; outputs as one tar.
- Keys only in `.env` / `~/.kaggle`; never in the transcript.

