# Plan

Written 2026-09-20 after the audit in `docs/POSTMORTEM.md`. Supersedes the
"Next" list in `STATE.md`, which was a queue of tasks rather than a sequence
with reasons.

The project is two days old. There is no deadline pressure; there was a
sequencing failure. This plan fixes the sequencing.

---

## The one thing that is actually wrong

Every measurement points at the same place from a different side.

**From the eval:** the hard eval asks for a 16-minute explainer and gets 16.3
seconds — a length ratio of 1.76%, with 5.0 play calls, flat across 81 tasks.
Concept coverage is 8.8% and drifting down. The verdict written at the time:
*"the model is not getting the subject wrong; it is answering a different,
much smaller question."*

**From the corpus, measured for the first time during this audit:**

| | corpus | gold |
|---|---|---|
| mean `self.play()` calls | 10.6 | 17.8 |
| share of visual vocabulary that is text | 41% | 36% |
| uses `ValueTracker` | **1.3%** | 9.1% |
| uses `always_redraw` | **0.9%** | 9.1% |
| asserts its own numbers | 2.7% | 95.5% |

`ValueTracker` and `always_redraw` are what make an animation continuous
rather than a sequence of slides. They are effectively absent. The three
commonest objects are `FadeOut`, `Text` and `Write`.

**These are the same fact.** The model answers a smaller question because the
corpus asks a smaller question. The render gate filters for *executes*, which
is necessary and is not sufficient. Nothing in the pipeline has ever filtered
for *animates*, and nothing has ever filtered for *explains*.

The consequence for planning is sharp: **generating more rows from the same
teacher with the same prompts makes this worse, not better.** More slideware
is more slideware. Any plan whose first move is "produce more data" is
building on the defect.

---

## Sequence

Four phases. Each ends in a number that decides whether the next one is worth
doing. Nothing runs in parallel with something it could confound.

### Phase 0 — Foundations (mostly done)

*Purpose: make the pipeline honest before trusting any measurement from it.*

| | state |
|---|---|
| Process groups killed properly | done — `45c98d6`, tested |
| Gold export reaches the dataset | done — `bffb1dc`, tested |
| Generated rows have unique ids | done — `bffb1dc`, tested |
| Static rows out of training | done — `bffb1dc`, tested |
| Ledgers distinguish killed from failed | done — `796fc4d` |
| Every artefact catalogued | done — `796fc4d` |
| Docs re-derived from data | done — `83ff933` |
| Test suite | done — `90e20bb`, 22 tests |

Remaining, small:

- **`forge doctor`** — one command that re-derives every number in `STATE.md`
  and diffs it against the file. The staleness in D1 was not caused by
  carelessness; it was caused by the check being manual.
- **`add_manim_import` lint rule.** 103 failed corpus rows parse cleanly,
  declare a `Scene` subclass, have no manim import, and failed with a
  NameError. Zero *passing* rows lack the import. This is mechanical, costs
  no API quota, and is the one remaining thing `regate` could recover.
  Expected yield: up to 103 rows (5.4% of the failed set). Cost: ~20 minutes
  of CPU. **This is the measurement that tells us whether the number is real.**

*Exit condition: `pytest` green, `forge doctor` clean, regate run once against
the new rule with the recovery counted.*

### Phase 1 — Train something, finally

*Purpose: the entire project rests on the claim that a render-gated corpus
beats a scraped one. That claim has never been tested. Every number on record
is the untuned model plus inference scaffolding.*

This is the single most overdue item and it is not close. It was blocked on
"the corpus is not large enough yet", which was never a threshold anyone
defined, and which the audit shows was measuring the wrong thing anyway.

- Train **once**, on the current 2,989-example mix, on Kaggle. `kaggle/01_sft.py`
  exists; the packaging script exists.
- Evaluate on **both** existing benchmarks, unchanged: the 100-row single-scene
  benchmark (untuned: 93% at rounds=4) and the 81-task hard eval (untuned: 85%
  render, 1.76% length ratio, 8.8% coverage).
- **One variable.** Nothing else changes — not the repair budget, not
  retrieval, not the system prompt. That rule has been broken three times
  already and each break cost an experiment.

The interesting outcome is not "did the render rate go up". It probably will,
slightly, and it is the least important number. The question is whether the
**length ratio and concept coverage move at all.** If they do not — and the
corpus analysis above predicts they will not — then Phase 2 is the whole
project and Phase 1 was the experiment that proved it.

*Exit condition: two eval numbers against two baselines, recorded in
`RESULTS.md` with the mix hash that produced them.*

### Phase 2 — Make the gate judge animation, not just execution

*Purpose: this is the real work, and everything before it is preparation.*

The render gate answers "did this produce a video file". A second gate has to
answer "is this an explanation". Not with a model-judge first — with
structure that can be measured and argued about:

- play calls and their spacing over the scene's duration
- presence of continuous constructs (`ValueTracker`, `always_redraw`,
  `.animate`, updaters) versus discrete `Write`/`FadeOut` pairs
- ratio of text mobjects to geometric ones
- whether any displayed number is computed rather than typed
- beat structure and declared-vs-actual duration

The 44 gold scenes are the labelled positive set — that is what they are
*for*, and it is the first time they would be used as anything but training
weight. Anything the measure scores below the gold floor is not a training
row, whatever the render gate said.

Then the corpus gets rebuilt against it, and the synthetic prompts get
rewritten to ask for the thing the measure rewards.

*Exit condition: a measure that separates the 44 gold scenes from a random
sample of corpus rows with the gap visible without squinting, and a rebuilt
mix whose profile sits between the two.*

### Phase 3 — Length and structure

*Purpose: the 1.76% problem, which Phase 2 does not address.*

A 16-minute explainer is not one scene. The `decompose` task type was written
for exactly this and has never been evaluated. This phase is: plan → several
scenes → render → concatenate, with the beat system already in place as the
unit.

Deliberately after Phase 2, because a chapter made of eight slideware scenes
is worse than one slideware scene, not better.

### Phase 4 — Serve it

`docs/ARCHITECTURE.md` is written and costed against measured latency:
separated inference and render services, ~$0.007 per video, Modal's free tier
covering roughly 4,000 videos a month. Nothing is built.

Last on purpose. There is no point serving a model that answers a smaller
question than the one asked, and the architecture does not change based on
what Phases 1–3 find.

---

## What is explicitly not happening

- **No always-on daemon pool until Phase 2.** The pool's job was to keep the
  API busy producing rows, and more rows of the current kind is the thing this
  plan says not to do. `forge_run.py` stays fixed and stays stopped.
- **No new gold scenes until Phase 1 is measured.** 44 of 61 is enough to
  train on and enough to use as the Phase 2 positive set. The remaining 17 are
  Tier 3, they cost hours each, and they are not on the critical path for any
  question currently open.
- **No more corpus generation** until there is a measure that can tell whether
  a generated row is worth keeping.
- **No parallel phases.** Every confound on record came from running two
  things at once and attributing the result to one of them.

---

## Standing rules this plan is built on

- One variable per experiment. Broken three times; each break cost a result.
- Preserve every artefact, failures included. (`data/raw/` is empty — the
  3,753 rows the dedupe discarded were never kept, so the dedupe's cost
  cannot be audited without re-downloading. Worth fixing the next time
  ingestion runs.)
- Busy is not the metric. Produced-and-arrived is the metric.
- Do not ship something that merely renders. That is now also the definition
  of the Phase 2 gate.
