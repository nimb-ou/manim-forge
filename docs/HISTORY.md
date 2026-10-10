# History: Sep 19 → Oct 8, 2026

What was built, what each phase measured, every mistake that cost time, and
what finally worked. Written Oct 8 from the git log (498 commits), RESULTS.md,
POSTMORTEM.md, PLAN.md and REASSESS.md, so that the next decision is made
knowing all of it. Numbers are the ones recorded at the time.

## The goal

Type a sentence, get a short 3Blue1Brown-style animation, made by a local
model on a 16 GB Mac. Non-commercial, built in the open.

## The phases

| dates | phase | what it produced | what it measured | verdict |
|---|---|---|---|---|
| Sep 19–20 | Harness and corpus | render harness with failure classes; 3,680 scraped scenes → 1,798 verified; 667 synthetic; 44 hand-written gold scenes; repair loop; retrieval | untuned 7B single scene: 40% → **93%** render with prompt + repair + retrieval | **worked** — inference-time work, no training |
| Sep 21–22 | First SFT (run 17), raw Manim | adapter on 3,010 rows | hard eval: coverage 8.4% → 18.8%, length ×2, beats 0 → 4.5; but render 93% → 77% | **mixed** — a fine-tune taught structure and cost reliability |
| Sep 23–24 | Two-stage pipeline | planner (arc in windows) + per-beat coder; planner v2, v3 | untuned split works; planner v2 loops; v3 sampled writes long arcs | **worked as plumbing**, quality untested by eye |
| Sep 25–27 | The Forge kit | 64 animation blocks, fuzz-tested; kit coder v5–v7; GRPO | beats that draw something 34% → **66%** (kit v5); GRPO "88%" was gamed (same picture five times) → 47% on the honest measure | **kit worked; GRPO did not** |
| Sep 28–29 | Judges and critics | Gemini vision judge, local Qwen3.5-4B judge, critic filtering | held-out 37–40% (v6), short 45–51% | measuring started — late |
| Sep 30–Oct 4 | Teacher batches | **127 commits** of hand-written teacher scenes, ~941 scenes, every one looked at and maths-checked | nothing measured in between | **the spiral** (below) |
| Oct 2–4 | — | the Mac slept **65 h** | — | lost |
| Oct 4–6 | kit v8 + planner v5 | SFT on the bigger data | Gemini held-out: v8 **44%** vs v6 **43%** | **400 more scenes bought one beat** |
| Oct 7 | Reassess + new approach | stopped data growth; built the one-shot generator over the scene library, narration, arithmetic and layout checks, an in-scope test set, a by-eye rubric | in-scope by eye: shipped app **0 good of 20**; one shot, untuned **12 of 16** | **the turn** |
| Oct 7 | One-shot v1 SFT | retrieval-augmented fine-tune, 1,731 rows | **6 good of 20** — worse than untuned | fine-tune hurt again |
| Oct 8 | Run 2: base models | same engine for all | Qwen3.5-9B untuned **13 / 20** in-scope, **8 / 20** held-out; 7B untuned 10; 7B fine-tuned 6 (held-out 3) | **v1.0 engine chosen** |

## What worked, in order of impact

1. **The Forge kit.** A small model writing `apply_matrix(stage, p, M)`
   instead of raw Manim produces clean frames. Everything good since Sep 26
   sits on it.
2. **The hand-written scene library**, used as *retrieved examples* rather
   than as training data. The same ~940 scenes that moved the fine-tune by
   one beat took the untuned model from 0 to 12 good of 16 when shown in
   the prompt.
3. **Writing the whole scene in one call.** The two-stage pipeline wrote
   each beat seeing only the earlier beats' intents; its scenes drifted
   off-topic for eight beats. One call sees its own code and stays coherent.
4. **A stronger base model.** Qwen3.5-9B (March 2026) does the arithmetic
   the 2024 7B got wrong; held-out 8 good against 3.
5. **Grading by eye with a written rubric**, beside every judge.

## What did not work

- **Every fine-tune since the first, judged by the end result.** Run 17
  traded render rate for structure; kit v5→v8 plateaued (43–44% Gemini);
  GRPO gamed its reward; the one-shot SFT lost the base model's arithmetic
  (12 → 6 good). The narrow-SFT pattern is consistent: it teaches format and
  costs reasoning.
- **More training data of the same kind.** ~400 extra teacher scenes: +1
  point.
- **The local vision judge** (Qwen3.5-4B): +14 points where Gemini saw +1;
  it nearly sent round 2 into self-training on a false gain.
- **Per-beat metrics.** "Visual beats", "relevant share" and the per-beat
  judge all rewarded scenes that a person would call wrong.

## Mistakes, and what each cost

| mistake | cost | lesson (and where it now lives) |
|---|---|---|
| Data silently lost between pipeline stages (32 gold scenes never reached the dataset, 2 of 3 synthetic files never read, ...) | weeks of data and runs, Sep 19–21 | compare both ends of every step (POSTMORTEM.md; `forge.doctor`) |
| Numbers copied forward instead of re-measured; no tests at first | wrong decisions, Sep 20–22 | same-day controls; CI-gated commits (memory: ci-gate-commits) |
| Infrastructure crowded out the goal (supervisor, autopilot, rotations) | days | build for the next measurement, not the next week |
| **The teacher-batch spiral**: a batch every 30-minute check for five days with no measurement able to say whether batches helped | ~5 days, ~127 commits | measure before adding more of anything (PLAN.md week rules) |
| Trusting the lenient local judge | nearly a wasted training round | decisions by eye + the strict judge only |
| Held-out topics leaked into training (CLT, chain rule, Monty Hall, dice totals) | inflated held-out numbers through v6 | `heldout_guard.py`, widened twice |
| The Mac slept 65 h with everything queued | 2.5 days | caffeinate tied to the job (memory: sleep-guard) |
| The Mac ran on battery: MLX ~50× slower, a whole night lost; zsh `&` jobs niced | ~10 h, Oct 7 | check `pmset -g batt`; launch via bash; move GPU work to Kaggle (memory: mac-power-and-priority) |
| Kaggle CLI hangs and 20-files-an-hour downloads | ~6 h, Oct 7 | every call under a deadline; outputs leave as one tar |
| Optimising per-beat proxies instead of whole scenes | the reason v6–v8 looked better than they were | whole-scene grading, `judge_scenes.py` + eye |
| Training first, trying the untuned model with good context last | about two weeks | always run the no-training baseline with the best prompt first |

## Where it stands, Oct 8

- **Engine:** one shot, Qwen3.5-9B untuned (MLX 4-bit, ~36 s to write a
  scene on the Mac), two retrieved scenes, kit reference, arithmetic and
  layout checks, best of 2, Kokoro narration. About 50 s a narrated scene.
- **Quality, by eye:** in-scope 13 good / 5 partial / 2 bad; held-out
  8 / 8 / 4.
- **The app** (`forge.serve`) defaults to it. The README describes it.
- **Not done:** the app run end to end over HTTP and timed; the gallery;
  tagging v1.0; a release of anything to Hugging Face (needs Nimit's OK, and
  v1.0 has no adapter to release — the scene library and kit are the
  artefacts).

## Oct 8 → 10: v1.5

Nimit tried v1.0 ("how gradient descent works?"): the kit wiped the axes,
the ball sat off the curve, nothing explained why. What followed, and what
it measured (all by eye; RESULTS.md has the tables):

| phase | result |
|---|---|
| 30 real-world dev requests | v1.0 17 / 10 / 3 (lenient grading, see below) |
| lesson plan first | +2 good; kept |
| thinking mode | 0 of 9 scenes written; dropped |
| kit repairs counted, critique-and-rewrite | beats a blind second sample 3–1 like-for-like; kept |
| self-check of its own numbers | unreliable both ways; off |
| LoRA on its own 64 best scenes | dev +3, fresh −2: nothing; published, off |
| **fresh test, blind** | **v1.5 10 / 7 / 3 against v1.0 7 / 11 / 2** |

New mistakes, and what they cost:

| mistake | cost | lesson |
|---|---|---|
| Grading drift: the same 30 scripts graded 19 good on Oct 8 and 12 on Oct 9 | a day of believing the plan helped more than it did | grade comparisons in one sitting, blind (`scripts/blind_pairs.py`) |
| A library change and an engine change measured in one run | a confounded result; a control run (~1 h) | one change per run, or a control |
| A Kaggle kernel named like its dataset (409 Conflict) | 20 min | distinct slugs |
| `prepare_model_for_kbit_training` upcasting Qwen3.5's 2B embedding parameters to fp32 | three OOM smoke runs | upcast only what trains; logits only where there are labels |
| A commit chained with `echo`, so a failing pytest committed | one red CI run | `r=$?; [ $r -eq 0 ] && git commit` (memory: ci-gate-commits) |
| The v1.0 worktree had no `.venv`, so every render "failed" | 40 min | check the first result before leaving a run |

What held up again: the untuned model with good context, checks that speak
to it in words, and grading by eye. What did not, again: training it.
