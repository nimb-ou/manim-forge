# Results

Every measurement, in order, with what changed between them. Benchmark data
lives under `data/bench/` which is gitignored (it is large and regenerable), so
this file is the durable record.

**Benchmark**: ManimBench v1 held-out test split. Render Success Rate — the
fraction of generated scenes that execute and produce video. Greedy decoding
(temp 0) so runs are reproducible. Environment failures excluded from the
denominator.

---

## Baseline progression — untuned Qwen2.5-Coder-7B-Instruct-4bit

| n | RSR | first try | rescued | what changed |
|---|-----|-----------|---------|--------------|
| 20 | 40% | — | — | bare system prompt, no repair |
| 20 | 70% | 13 | 1 | + "always animate with self.play() and end with self.wait()" in the system prompt, + repair loop |
| 20 | 85% | 14 | 3 | + few-shot retrieval over verified scenes |
| **100** | **83.0%** | 75 | 8 | same configuration, full split — the first defensible number |
| **100** | **89.0%** | 77 | 12 | + repair escalation, + stdlib import lint |
| **100** | **93.0%** | 77 | 16 | repair budget 2 rounds -> 4 |

Nothing above is trained. Every point came from inference-time work.

### Attribution

- **One sentence** in the system prompt was worth ~25 points, by eliminating
  the whole `empty_render` class. The repair loop, far more machinery, added 5.
- **Retrieval** added ~15, and made repair three times more effective —
  examples behave nothing like documentation, which is a measured failure mode
  for small models.
- **Escalation** added ~4 (rescued 8 → 12). **Import lint** added ~2
  (measured at +1 in isolation).

### The repair budget: the one clean experiment so far

Raising `max_rounds` from 2 to 4 took the rate from **89.0% to 93.0%**. Only
the round count changed, and this comparison does not depend on trusting that:
the two runs agree on **77 first-try passes**, and within the rounds=4 run
exactly **4 successes used a round beyond the second** —

```
rounds used by the 93 successes:  0:77   1:6   2:6   3:3   4:1
the 4 late rescues, by error history:
  api_misuse, api_misuse, api_misuse            -> none
  name, api_misuse, api_misuse                  -> none
  name, api_misuse, api_misuse, api_misuse      -> none
  name, api_misuse, api_misuse                  -> none
```

Truncate that run at 2 rounds and it scores 89 — the control's exact number.
The effect is recoverable *inside* a single run, which is what the earlier
confounded experiments could not offer.

Every late rescue is an `api_misuse` chain ending in a pass, which is the
escalation prompt working: after two identical error kinds it tells the model
to delete the offending construct and rebuild with the simplest objects
available. It needs a third and fourth attempt to land.

**Not comparable:** wall clock. Rounds=4 finished in 41.5 min against the
control's 46.6, which is backwards for strictly more work — the control shared
the machine with two generation daemons. Time here measures load, not method.

### Confounds — stated, not hidden

Two comparisons here changed more than one variable:

1. The 70% run altered the system prompt *and* introduced the repair loop.
   Attribution survived only because rounds-used was logged per trial.
2. The 89% run was meant to isolate escalation; the import lint went in as
   well. It decomposes because first-try and rescued-by-repair are recorded
   separately, but that is luck.

For inference experiments this costs ten minutes to redo. For training runs it
costs a week of compute and yields a number that cannot be trusted.

### What is left

7 failures at 93%: **5 `api_misuse`**, 1 syntax, 1 unknown. `api_misuse` is
the model calling a real Manim class with arguments that do not exist. It has
now survived the system prompt, introspected API briefings, retrieval,
escalation, linting, and a doubled repair budget — each of which removed some
of it and none of which removed the rest.

That is the residue fine-tuning has to attack. It is also the reason the
corpus is render-gated: a model that has seen 1,760 examples of constructors
being called correctly is the only intervention left that acts on the model's
priors rather than on its output.

---

## Corpus

*Counts re-derived from the data files 2026-09-20; the previous version of
this table was stale by 40 gold scenes.*

| source | rows | verified | note |
|--------|------|----------|------|
| 5 public datasets, deduplicated | 3,680 | 1,760 (47.8%) | 7,433 raw rows; half were duplicates |
| Synthetic — topics | 186 | 138 | 115 topics across 12 domains |
| Synthetic — continuous daemon | 429 | 399 | `stream.jsonl` |
| Synthetic — 3b1b narration | 175 | 130 | his words as the prompt |
| Gold — hand-authored | **44** | 44 | 182 beats; weighted 6× at training |
| 3b1b transcripts | 5,825 segments | — | 151 videos, subtitles only |

Training mix as actually written: **2,989** train / 125 valid, of which 1,437
gated · 1,294 synthetic · 258 gold. Rows with zero `self.play()` calls are
excluded (243 of them). Until 2026-09-20 this mix was 1,995 examples with gold
at 3.6% and 228 static rows included — see `docs/POSTMORTEM.md` §A, §B.

### What the corpus teaches, measured

Never measured until the audit, and the most consequential number here.
Synthetic training rows against the hand-written gold scenes:

| | corpus | gold |
|---|---|---|
| mean `self.play()` calls | 10.6 | 17.8 |
| share of visual vocabulary that is text | 41% | 36% |
| uses `ValueTracker` | **1.3%** | 9.1% |
| uses `always_redraw` | **0.9%** | 9.1% |
| asserts its own numbers | 2.7% | 95.5% |

`ValueTracker` and `always_redraw` are what make an animation continuous
rather than a slideshow, and they are effectively absent. The commonest
objects are `FadeOut` (11,543), `Text` (10,813), `Write` (10,200); `Dot`
appears 1,732 times.

This is the hard eval's finding — *"the model answers a much smaller
question"* — arriving independently from the data side. The gate filters for
**executes**. Nothing filters for **animates**, and nothing filters for
**explains**.

Dedupe is AST-structural, so renamed variables and reflowed whitespace collapse
together. `generaleoley` (1,622 rows) contributed **zero** new rows — it is
entirely contained in `thanhkt`.

---

## Hard eval — a whole explainer from one line

81 real 3Blue1Brown video titles as one-line prompts. Deliberately much harder
than the benchmark, and expected to score poorly at first: the point is a
measurement that keeps moving as the model improves at the actual job rather
than one that saturates on single scenes.

Scored on render success, duration against the real runtime, beat count, and
concept coverage — whether the generated narration mentions the terms the real
video spends its time on.

### Full run — untuned model, all 81 tasks, repair budget 4

| metric | value |
|--------|-------|
| render success | **85%** (69/81) |
| mean duration | **16.3s** |
| real videos average | **16 min** |
| length ratio | **1.76%** |
| concept coverage | 8.8% |
| mean play calls | 5.0 |

Render success is 85% here against 93% on the single-scene benchmark, and the
drop is the point: the same model, the same repair budget, asked for something
eight times longer. The length ratio is the number that matters. **1.76%** —
the model answers a sixteen-minute chapter with a sixteen-second clip, and
does it consistently: the mean is 5.0 play calls, which is one beat of a
scene, not a structure.

Note the direction of the two gaps. Coverage is 8.8% and falling slightly
against the 15-task baseline, while length is flat. The model is not getting
the subject wrong; it is answering a different, much smaller question. No
amount of inference-time repair fixes that, because nothing the repair loop
sees is broken.

### Baseline — untuned model, 15 tasks

| metric | value |
|--------|-------|
| render success | 80% (12/15) |
| mean duration | **15.9s** |
| real videos average | **11 min** |
| length ratio | 2.55% |
| concept coverage | 10.6% |
| mean play calls | 5.2 |

The diagnosis is more useful than the score. Best case — *"Matrix multiplication
as composition"* — reached **31.8% coverage in 34 seconds**, matching `matrix`,
`transformation`, `linear`, `multiplication`. The model produces **topically
relevant but drastically too short** output: it answers with a clip where a
chapter was asked for.

So the gap on this eval is length and structure, not subject understanding.
That is what the `decompose` task type targets — breaking a large request into
several scene-sized ones — and this measurement is the reason to believe it
matters rather than an assumption that it might.

---

## Free-tier reality

The documented Gemini free tier (1,500 requests/day for Flash) does not match
what the key actually delivers. Measured across a full day:

| | |
|---|---|
| total calls before exhaustion | **~944** |
| models exhausted | all 6 in rotation |
| reset | daily, Pacific midnight |

Roughly **900–1,000 generations per day**, not per model. At ~95% verified
yield that is ~900 scenes/day, so a 10,000-scene corpus takes about ten days.
Rotation across six models still matters — it is what reaches that ceiling
instead of stopping at one model's share — but it does not multiply it.

Raising concurrency does **not** help and actively hurt: 5 workers produced
~10 tasks/min while 8 produced ~5.7, because extra workers front-load the
exhaustion and then everything waits.

---

# Phase 1 — the first trained adapter · 2026-09-22

**Run 17**, Qwen2.5-Coder-7B-Instruct, QLoRA r=16 alpha=32 on all seven
projections, 1 epoch over 3,010 rows, mix `fbb31a386b28f197`.

Every number below is against a **same-day control** run with the
byte-identical command, differing only in `--adapter`. The control
reproduced the recorded baseline exactly — 93% / 77 first-try / 16 rescued
on the benchmark, 85% / 16.3s / 1.76% / 8.8% on the hard eval — so the
comparison is one-variable, which this project has failed three times
before.

## Single-scene benchmark (100 prompts, repair rounds 4, retrieval)

| | control | run 17 |
|---|---|---|
| render success | 93.0% | **77.0%** |
| passed first try | 77.0% | 73.0% |
| rescued by repair | 16.0% | **4.0%** |

## Hard eval (81 real 3Blue1Brown titles)

| | control | run 17 | |
|---|---|---|---|
| render success | 85.2% | **71.6%** | −13.6 pts |
| concept coverage, all trials | 8.4% | **18.8%** | **2.2×** |
| concept coverage, rendered | 8.8% | **16.8%** | 1.9× |
| mean duration, rendered | 16.3s | **31.0s** | 1.9× |
| length ratio | 1.76% | **3.47%** | 2.0× |
| mean play calls | 5.4 | **11.3** | 2.1× |
| mean beats | **0.00** | **4.54** | from nothing |

## What this says

**The premise held, and the plan's prediction was wrong.** `docs/PLAN.md` §2
predicted a fine-tune on this mix would not move length ratio or concept
coverage. Both roughly doubled, duration doubled, play calls doubled, and
beat structure appeared where the untuned model produced none at all. A
render-gated corpus does teach longer, denser, better-covering explanations.

**The render drop is the price of that, not a failure of it.** Paired prompt
by prompt: first-try 77 → 73 (−4), but repair rescues 16 → 4 (−12). Repair
effectiveness fell 70% → 15%, three quarters of the loss.

`failure_anatomy.py` says why repair stopped working. Run 17's failures are
**80% longer than its own passes** — 23 lines against 13, 19 calls against
11. Reading them: where the control wrote `Dodecahedron()`, run 17 hand-built
a `Polyhedron` from vertex coordinates; where the control drew a
`SurroundingRectangle`, it tried `table[0, :].set_color(...).animate`. All 23
remaining benchmark failures are `api_misuse`, with the same error at every
one of four rounds.

**The corpus moved the model's ambition past its API competence.** Repair
cannot rescue that: the approach is wrong, not the line.

**Consequence for measurement.** Render rate and explanation quality trade
off through ambition, so render rate at a fixed repair budget is the wrong
headline — the product repairs until a scene renders, and a scene that took
three repairs is not worse than one that took none. It is a cost, not a goal.

**Not undertrained.** Eval loss plateaued at step 240 of 377 while train loss
kept falling (0.5604 → 0.5619 → 0.5611 → 0.5609 against 0.80 → 0.49), so more
epochs buy overfitting, not competence.

*Artefacts: `data/bench/{ctrl,tuned}100_n100_r4.json`,
`data/bench/{ctrl,tuned}81_n81.json`. Adapter `adapters/kaggle-sft/adapter`,
161,533,192 bytes, conversion verified by `scripts/verify_adapter_math.py` at
3.1e-04 relative error.*

---

# The planner adapter · 2026-09-23

First half of the two-stage split. 1,226 windows from 190 arcs (44 gold, 146
real 3Blue1Brown videos), 162 MB, conversion verified at 2.9e-04 relative
error.

Same prompt, same task, tuned against untuned:

**Tuned**
```
1. [18s] A beam of light, and a question -- A beam of light. What is it?
         It is not a wave, and it is not a particle...
2. [22s] A beam of light, and a screen -- Shine the beam on a screen, and
         it spreads out into a fuzzy blob...
3. [23s] A beam of light, and a slit -- Shine the beam through a slit...
```

**Untuned**
```
1. [0.0s]  -- Introduce the concept of light and its fundamental properties.
2. [2.0s]  -- Explain that light is a form of electromagnetic radiation.
3. [4.0s]  -- Describe the speed of light and its significance.
```

Three differences, all of them the point:

- **Durations of 18–23 seconds against 0, 2, 4.** The untuned model has no
  idea what a beat costs. Real 3Blue1Brown beats run about 26 seconds.
- **Real visual intents against an empty intent field.** The untuned model
  copied the word "intent" out of the format spec.
- **Narration in 3Blue1Brown's voice against a table of contents.**
  "Introduce the concept of" is not something anyone says out loud.

The 151 real narration arcs taught that, and they had been sitting unused in
`data/style/narration.jsonl` since the scrape.

**This is not yet the split working.** The coder is still untuned and
assembly is still where the pipeline fails. It means the planner half of the
premise holds: ambition expressed in text, in the right shape.

## Two-stage pipeline, untuned both halves

| | n=6 |
|---|---|
| assembled | 2/6 → 3/6 after the harness fixes |
| rendered | 1/6 → 0/6 |
| mean beats | 6.0 |

Run before either adapter existed, on purpose: if the machinery does not
work, no adapter rescues it, and "the split fails" and "the adapters are not
good yet" want opposite responses. Both failure modes turned out to be the
harness rather than the idea — beats truncated at 600 tokens destroying the
scenes they were concatenated into, and a coder inventing names because
nothing told it what was already in scope. Both fixed; neither number is
meaningful until the coder is trained.

---

# The coder adapter · 2026-09-23

9,504 beat→method rows from 1,071 scenes (182 gold beats at weight 6, the
rest decomposed from render-verified corpus scenes), `max_len` 768, split by
scene so no scene straddles train and validation.

**Early stopping fired, and here it was worth something.**

```
eval_loss  0.678  0.612  0.594  0.581  0.579  0.591  0.594  0.602
step          60    120    180    240    300    360    420    480
train loss 2.499 ──────────────────────────────────────────→ 0.408
```

Three evaluations of degradation after step 300 stopped the run at 480 of
1,188 planned steps, and `load_best_model_at_end` means the saved adapter is
step 300 rather than step 480 — 0.579 against 0.602. On run 17 the same two
settings were worth 0.0005 and I recorded them as insurance; here they are
the difference between the best checkpoint and one well into overfitting.

Collected, converted and verified without a person: `collect_adapter.py`
waited for the run, downloaded with retry, checked the weights read back,
converted to MLX and confirmed the delta survives at 3.2e-04.

---

# The split renders · 2026-09-23

Eight single-scene prompts, 12-beat plans, planner v1 throughout.

| coder | assembled | rendered |
|---|---|---|
| untuned | 3/6 | 0/6 |
| v1 | 6/8 | **0/8** |
| v2 | 5/8 | **3/8** |

The difference between v1 and v2 is entirely training data, and it is the
first time the two-stage pipeline has produced a scene that renders.

v1 was trained on 182 gold beats of which 139 call helper methods — `panel()`,
`arrow()`, `grid()` — that a bare assembled `Scene` does not have, and on two
incompatible conventions for carrying state between beats (`self.x` in gold,
plain locals in the 8,649 decomposed corpus rows), with gold weighted six
times. It learned the minority convention and the absent helpers. Every
assembled scene died on `api_misuse`.

v2 trains on gold localised to plain variables and filtered to the 43 beats
the runtime can satisfy. 258 gold rows rather than 1,092.

**What is left is one failure, three times.** Three of eight fail assembly on
names no beat defines — `dot, square, triangle`, `axes, dot, function_graph`,
`dot1, dot2, dot3`. A later beat uses objects an earlier beat was supposed to
create and did not. That is the failure the split introduces by construction,
it is caught before a render rather than in a render log, and it is the next
thing to fix rather than a reason to abandon the split.

Not yet comparable to Phase 1: these are single-scene prompts at 12 beats,
not the 81 titles. The hard-eval comparison waits for the assembly failures,
because a pipeline that drops three of eight scenes would report a coverage
number computed on the five that survived.

## The coder alone · 2026-09-23

`scripts/beat_eval.py` renders each held-out beat of the coder's validation
split on top of the *reference* bodies of the beats before it — the coder
measured with the planner held out.

| adapter | beats render | api_misuse | undefined names | unparsable | excluded (reference fails) |
|---|---|---|---|---|---|
| coder v2 | **179 / 193 (92.7%)** | 8 | 5 | 1 | 20 |

Against 2–3 of 8 end-to-end, this moves the problem. Given correct earlier
beats, the coder writes a beat that renders nine times in ten; the
end-to-end losses are in composition — planner v1 repeats one intent across
up to nine beats, never writes END, and when the beat that should build an
object is dropped, every later beat that uses it fails the scene. Coder v3
is measured against 92.7%, and planner v2 is now the larger lever.

## Can Kaggle render? · 2026-09-24

GRPO's reward is "the beat renders", so the trainer needs Manim beside the
GPU. Probed on a Kaggle CPU session (`kaggle/probe_render`, no GPU quota):

| step | result |
|---|---|
| apt: cairo, pango, ffmpeg, texlive-latex-extra, dvisvgm | ok, 66 s |
| pip install manim==0.21.0 | ok, 42 s |
| render Circle + Text + MathTex, -ql | ok, 5.7 s cold, **2.6 s warm** |
| four renders in parallel (a GRPO group) | all ok, **5.1 s** |

Two minutes of setup and about 1.3 s per completion at group size four. The
reward is not the bottleneck; generation on a T4 will be.

## Planner v2 loops · 2026-09-24

Plan-only on eight hard titles, cap 48: **48.9 beats per plan, END never
written, 38–51 repeated intents per plan** ("two dice, one dot
highlighted" fifty times). The opening beats are good — "two vectors, a
line of all combinations" → "three vectors, a plane" → "four vectors, a
whole space" is a real arc — and then greedy decoding, fed its own output,
settles into a loop.

Two causes, both addressed: half of the 252 synthetic arcs it trained on
padded a thirty-beat quota by repeating one intent (planner v3 trains on
arcs cut at their first repeat, teacher told to END instead of padding),
and the driver had no stopping rule besides END (a repeated intent now
ends the plan).

## Split renders after the harness fixes · 2026-09-24

Same eight tasks, same planner v2 plans (greedy, stopped at the first
repeated intent, so 4.8 beats mean), `--salvage`, after fixing three
harness faults (an unparsable beat cost the whole scene; `self.mobjects`
flagged as undefined; a repair that added unknown names was kept):

| coder | assembled | rendered | beat eval (alone) |
|---|---|---|---|
| v2 | 6/8 | **5/8** | 92.7% |
| v3 | 6/8 | 3/8 | 92.7% |

Coder v3 — one system prompt, rows that break the CONSTRUCT rule removed —
is not better on either measure. At n=8 the gap is two scenes and could be
noise; it is not evidence *for* v3. v2 stays the default. The step from 3/8
to 5/8 is the harness, not a model.

## Split on the hard eval · 2026-09-24

First 24 of the 81 hard titles, planner v2 + coder v2, `--salvage`, plans
stopped at the first repeated intent:

| | rendered | coverage (all) | coverage (rendered) | length ratio |
|---|---|---|---|---|
| **split p2+c2** | **13/24** | 4.5% | 5.2% | 1.38% |
| Phase 1 control | — | 8.4% | — | 1.76% |
| Phase 1 run 17 | — | 18.8% | — | 3.47% |

The split renders — more than half the hard titles, where Phase 1's
failures were overwhelmingly render failures — but what it renders is
tiny: **3.8 beats a plan**, because planner v2 loops and the loop is cut at
its first repeat. So on the plan's own terms (*the split has to beat run
17*) it currently loses on coverage and length, and the reason is one
specific, identified defect in the planner rather than decomposition as
such. Planner v3 — trained on arcs cut at their first repeat, teacher told
to END rather than pad — is the test of that; if v3 plans run long without
looping, this table is re-run on all 81.
