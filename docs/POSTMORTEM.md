# Postmortem

Written 2026-09-20, after 27 hours and 62 commits, at the point where the
project was stopped because it was spiralling. Every claim here was re-derived
from the repo rather than recalled.

The purpose is not contrition. It is that several of these failures share a
shape, and the shape is worth naming before building anything else.

---

## The one-line summary

The project built a great deal of machinery for producing data and very little
for checking that the data arrived. Every defect below sat between a step that
worked and a step that consumed its output, and every one of them was invisible
because nothing ever compared the two ends.

---

## A. Defects that silently destroyed work

These are ranked by how much finished work they threw away.

### A1. 32 gold scenes never reached the dataset

`scripts/register_gold.py` exists to do the four bookkeeping steps that must
happen when a gold scene is finished, and its docstring says why:

> *…which is how a scene ends up rendered, committed, and invisible to training
> because nobody added it to GOLD.*

It added the scene to the `GOLD` list in `export_gold.py`, marked the
curriculum entry done, retimed the beats and ran the narration audit. It never
ran `export_gold.py`. So `data/gold/gold.jsonl` stayed at the 12 rows of the
last manual export while 44 scenes existed on disk.

**Cost:** 32 hand-authored scenes — the single most expensive artefact in the
project, several hours each — were finished, committed, audited, and absent
from every training run. At a gold weight of 6 they were 72 of 1,995 training
examples instead of 264.

**The shape:** a script written specifically to prevent a class of error,
which then committed that exact error, because writing it into the list and
writing it into the dataset are two steps and only one was automated.

### A2. Generated rows shared ten identities between 399 scenes

`CorpusRow.build` took a mandatory `index` used to mint the row id. The ingest
sources pass a real row number. `generate_forever.py` passed a literal `0` —
for every row it has ever written.

    data/synthetic/stream.jsonl: 399 verified rows
                                 399 distinct dedupe_keys
                                 399 distinct code bodies
                                  10 distinct ids
                                 369 of them called stream-narration:000000

Anything keyed on `id` — deduplication, joins, resume sets — discarded 97% of
that file. The training loader deduplicated by id, so it did.

**Cost:** 488 verified synthetic scenes, each one a full generate-render-repair
cycle against a rate-limited API, unused.

**The shape:** an identifier that was never checked for the one property an
identifier has. `dedupe_key` — a content hash — was sitting in the same record
the whole time and was correct.

### A3. Only one of three synthetic files was ever read

`prepare_training.py` defaulted `--synthetic` to `generated.jsonl` alone.
`stream.jsonl` (399 verified) and `from_narration.jsonl` (130 verified) — the
larger two, and the ones the always-on daemon was producing — were not in the
default mix.

**The shape:** a default written when there was one file, never revisited when
there were three. Nothing compared "files that exist" against "files that are
read."

---

## B. Defects that quietly degraded quality

### B1. 243 scenes with no animation were training the model not to animate

`forge/gate/export.py` says plainly:

> *The corpus is heavily static, and a model trained on it writes scenes that
> produce no video.*

`forge/train/prepare.py` then tried to act on that:

```python
reps = {"gold": 6, "synthetic": 2}.get(tier, 1)
if r["meta"].get("n_play_calls", 1) == 0:
    reps = max(1, reps // 2)     # "Static scenes ... halve them"
```

For a gated row `reps` is already 1, so this is `max(1, 0) == 1`. It changed
nothing. It could only ever have bitten gold and synthetic — the two tiers that
contain no static rows at all.

So 243 rows with zero `self.play()` calls entered training at full weight, each
paired with a system prompt that reads *"Always animate with self.play(...)"*.
Every one is a worked example of ignoring the instruction directly above it.

**The shape:** a mitigation that was written, reviewed, commented, and never
once measured against the thing it was supposed to mitigate. One line —
counting the rows it removed — would have shown it removing none.

### B2. The training file could not be audited

`prepare.py` wrote `{"messages": ...}` and dropped `meta`. Asking "how much of
this is gold" required re-deriving the mix from the inputs and trusting that
the derivation matched what was written. During this audit it did not: the
answer was 3.6%, not the ~15% the weighting implied.

### B3. What the corpus actually teaches, measured

This was never measured before today, and it is the most important number in
the audit. Comparing the 1,995-row training set against the 44 hand-written
gold scenes on the same metrics:

| | synthetic corpus | gold scenes |
|---|---|---|
| mean `self.play()` calls | 10.6 | 17.8 |
| rows with zero play calls | 228 | 0 |
| share of visual vocabulary that is text | 41% | 36% |
| uses `ValueTracker` | **1.3%** | 9.1% |
| uses `always_redraw` | **0.9%** | 9.1% |
| asserts its own numbers | 2.7% | 95.5% |

`ValueTracker` and `always_redraw` are the two constructs that make an
animation continuous rather than a slideshow. They are effectively absent. The
most common objects in the corpus are `FadeOut` (11,543), `Text` (10,813) and
`Write` (10,200); `Dot` appears 1,732 times.

A representative row, chosen by taking the first one in the file:

> **prompt** — *Dominating Sets: Unraveling Complexity in Hypergraphs…*
> **code** — a `Text` title, a `Text` reading "Welcome to the world of
> combinatorics!", then more `Text`.

It renders. It passes the gate. It teaches nothing.

**This is the same finding the hard eval reported** — *"the model answers a
much smaller question"* — arriving from the data side. The render gate filters
for *executes*, which is necessary and is not sufficient, and nothing in the
pipeline has ever filtered for *animates* or *explains*. That gap is the
project's central open problem, and it is not fixed by any change in this
document.

---

## C. Defects that burned the machine

### C1. `subprocess.run(timeout=)` kills one process, not the group

In `forge/harness/render.py`:

```python
proc = subprocess.run(
    cmd, ..., timeout=self.timeout,
    # New process group, so a timeout kills ffmpeg children too
    # rather than orphaning them to spin for the rest of the run.
    start_new_session=True, ...)
```

The comment describes behaviour the code does not have. CPython's timeout path
calls `process.kill()` on the single pid. `start_new_session=True` puts manim
in its own group that nothing ever signals — so it makes the orphaning *more*
certain, not less. Every render that hit the wall clock left ffmpeg and LaTeX
running.

### C2. Escalating to SIGKILL "only if still alive" never escalates

In `forge/orchestrator.py`:

```python
os.killpg(pgid, 15)
time.sleep(2)
if self.alive():          # <- the direct child
    os.killpg(pgid, 9)
```

A generation daemon exits promptly on SIGTERM. Its `ProcessPoolExecutor`
workers and the renderers those workers launched do not. The check asks about
the parent, sees it gone, and skips the sweep that would have caught the
children. Twenty-four orphaned workers accumulated this way across a handful of
restarts.

**C1 and C2 together are the load average of 99 on ten cores.** Both are now
unconditional `killpg` of the group, and both are tested with a child that
deliberately spawns a grandchild and hangs.

### C3. Two jobs rendering the same 44 scenes

`gold-verify` rendered every gold scene at 480p15 while `showcase` rendered the
same 44 at 1080p60. Six manim processes deep, for a result the second job
already proved. Removed during the session; recorded here because the roster
had no notion of what a job overlapped with.

### C4. A finished job restarted forever

`regate` had already run all 1,920 previously-failed rows against the current
lint rules — a rule fired on 122, and 8 recovered. With nothing left to do it
exited immediately, and the supervisor's `Health.DONE` branch started another
pass every thirty ticks. Restarting a job that finished in two seconds because
its queue was empty is a busy loop wearing the costume of a worker, and it is
indistinguishable, from the outside, from the productive work it was crowding
out.

### C5. Interrupted renders recorded as broken scenes

`render_showcase.py` recorded `ok: false` on any non-zero exit, including exit
by signal — that is, including *me killing it*. Eleven ledger failures: ten
were kills. Four of those scenes had already rendered successfully. The true
state was 19 of 44 rendered and exactly one genuine failure, not 23 and eleven.

**The shape, shared by C4 and C5:** the supervisor could not distinguish
*working*, *finished*, and *killed*. Every unclear case resolved toward doing
more work, so the machine stayed busy and the ledger filled with verdicts that
were not verdicts.

---

## D. Process failures

### D1. Numbers were copied forward instead of re-derived

`docs/STATE.md` claimed 10 gold scenes when there were 44, 89% when the
measured figure was 93%, and a verified corpus of ~2,300 when the honest count
is 1,759 unique gated rows. The file opens with *"If this file and the repo
disagree, the repo is right and this file is stale — fix it."* Nobody ran the
comparison.

Each individual number was true when written. Staleness is not a lie, but a
status document nobody re-derives is a status document that describes a
project that no longer exists.

### D2. There are no tests

62 commits. 18,863 lines of Python. `tests/` is an empty directory and `pytest`
is not installed. Every defect in sections A and B would have been caught by a
test asserting that the thing written equals the thing read.

### D3. Infrastructure crowded out the goal

The last stretch before stopping: five supervisor restarts in fifteen minutes,
chasing a 429 that turned out to be backpressure the SDK was already absorbing,
while orphaned renderers accumulated behind me. Meanwhile the corpus had been
ready to train on for hours and no fine-tune has ever been run.

The standing instruction was to keep API, CPU and context all busy. I optimised
for the *appearance* of that — four daemons running — over what they produced.
Two of the four were doing nothing (C3, C4), one was writing rows with
duplicate ids (A2), and the output of all of it was flowing into a training
file that was dropping two thirds of it on the floor (A1–A3).

**Busy is not the metric. Produced-and-arrived is the metric.**

---

## What changed

| | fix | commit |
|---|---|---|
| C1, C2 | `killpg` the group, unconditionally, both signals; tested | `45c98d6` |
| A1 | `register_gold.py` runs the export; gold ids keyed on module | `bffb1dc` |
| A2 | `index=None` → content-addressed id; loader keys on `dedupe_key` | `bffb1dc` |
| A3 | all three synthetic files read by default | `bffb1dc` |
| B1 | zero-play rows dropped, and the count reported | `bffb1dc` |
| B2 | `meta` written alongside `messages` | `bffb1dc` |
| C4 | finished jobs restart only if the last pass produced something | `796fc4d` |
| C5 | signal deaths record nothing; existing records annotated | `796fc4d` |
| D1 | `STATE.md` re-derived from the data | this commit |

Measured effect on the training mix:

```
train examples   1995 -> 2989
gold share        3.6% -> 8.6%
synthetic rows     276 -> 1294
zero-play rows     228 -> 0
```

Not fixed, and not fixable by any of the above: **B3**. The corpus teaches
slideware because that is what the corpus is, and the gate cannot tell the
difference. That is a design question, not a defect.
