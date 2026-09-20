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

| source | rows | verified | note |
|--------|------|----------|------|
| 5 public datasets, deduplicated | 3,680 | 1,760 (47.8%) | 7,433 raw rows; half were duplicates |
| Synthetic — topics | growing | ~94% yield | 115 topics across 12 domains |
| Synthetic — 3b1b narration | growing | ~91% yield | his words as the prompt |
| Gold — hand-authored | 4 | 4 | weighted 6× at training |
| 3b1b transcripts | 5,825 segments | — | 151 videos, subtitles only |

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
