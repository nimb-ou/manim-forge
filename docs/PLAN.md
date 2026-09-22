# Plan

Written 2026-09-20, after the audit in `docs/POSTMORTEM.md` and a round of
research into every service this project would depend on. Supersedes the
"Next" list in `STATE.md`, which was a queue rather than a sequence — which
is how four jobs ended up running at once and confounding each other.

The project is two days old. There is no deadline pressure; there was a
sequencing failure. Companion document: `docs/SETUP.md`, which is the list of
things only Nimit can do.

---

## 1. Where we stand

`python -m forge.doctor` for the live figures. At the time of writing:

| | |
|---|---|
| Scraped corpus | 3,680 deduplicated → **1,798 verified** (48.9%) |
| Synthetic | **667** verified |
| Gold scenes | **44 of 61** · 182 beats · all importing, audit clean |
| Gold at 1080p60 | **19 of 44** |
| Training mix | **3,010** train / 131 valid |
| Benchmark, single scene | **93%** render, repair rounds=4 |
| Hard eval, whole explainer | 85% render · 16.3s vs 16 min · 8.8% coverage |
| Models trained | **zero** |

The pipeline is complete end to end — ingest, gate, lint, repair, retrieve,
evaluate, and a local platform — and every number on record describes an
untuned Qwen2.5-Coder-7B with scaffolding around it.

---

## 2. What is actually wrong — revised 2026-09-22 by run 17

### 2.1 What the audit said, and how much of it survived

The audit diagnosed a corpus that asks a smaller question than the eval does:

| | corpus | gold |
|---|---|---|
| mean `self.play()` calls | 10.6 | 17.8 |
| text share of visual vocabulary | 41% | 36% |
| uses `ValueTracker` | **1.3%** | 9.1% |
| uses `always_redraw` | **0.9%** | 9.1% |
| asserts its own numbers | 2.7% | 95.5% |

Two of those rows do not mean what they were taken to mean. `ValueTracker`
at 9.1% of gold is **four scenes**; per scene it separates gold from corpus
at AUC 0.538, a coin flip. And gold is *more* text-heavy than the corpus in
absolute terms (16.4 mobjects against 7.7) — 3Blue1Brown scenes label
things. See Phase 2 for the full per-signal table.

What survived is the play-call gap and the grounding gap, and they were
enough.

### 2.2 The prediction, and the result

This section predicted that a fine-tune on the current mix **would not move
length ratio or concept coverage**, and that Phase 1 would be the experiment
proving Phase 2 was the whole project.

Run 17, at 60 of 81 hard-eval tasks (provisional — the same-day control is
still running):

| | baseline | run 17 |
|---|---|---|
| render rate | 85.2% | **71.7%** |
| coverage, all trials | 8.4% | **18.8%** |
| coverage, rendered only | 8.8% | **16.7%** |
| mean duration, rendered | 16.3s | **31.5s** |
| length ratio, rendered | 1.76% | **3.88%** |
| play calls | 5.4 | **11.2** |

**The prediction was wrong.** Every explanation-quality metric roughly
doubled. The render-gated corpus does teach longer, denser, more
concept-covering scenes. That is the project's premise, and it held.

### 2.3 The new binding constraint

Render rate fell 13.5 points, and the single-scene benchmark fell 93% → 77%.
That is not the corpus failing. It is the price of the thing the corpus
bought.

- Paired against the baseline: first-try 77 → 73, but repair rescues 16 → 4.
  Repair effectiveness 70% → 15%, three quarters of the drop.
- `failure_anatomy.py`: the tuned model's failures are **80% longer than its
  own passes** (23 lines against 13, 19 calls against 11).
- Reading them: where the control wrote `Dodecahedron()`, run 17 hand-built
  a `Polyhedron` from vertex coordinates. Where the control drew a
  `SurroundingRectangle`, it tried `table[0, :].set_color(...).animate`.

**The corpus moved the model's ambition past its API competence.** Repair
cannot rescue that, because the approach is wrong rather than the line —
which is exactly why four rounds changed nothing and why a fifth would not.

### 2.4 What this changes about steering

The project has been steering by render rate. Run 17 shows render rate and
explanation quality trading against each other, so it is the wrong headline:
the product repairs until a scene renders, and a scene that needed three
repairs is not worse than one that needed none. **The metric that matters is
explanation quality conditional on eventually rendering**, with render rate
as a cost, not a goal.

Consequence for planning, unchanged from the audit: generating more rows
from the same teacher with the same prompts is still not the first move. But
the reason has changed. It is no longer "the corpus teaches slideware" — it
is that the corpus already teaches the right ambition, and more of it buys
more ambition the model cannot execute.

---

## 3. The five phases

### Revised sequence, 2026-09-22

Run 17 reordered these. The premise held — a render-gated corpus does teach
longer, denser explanations — and the constraint moved to **API competence
on ambitious code**. The phases are re-prioritised against that, not against
the original diagnosis.

| was | now | why |
|---|---|---|
| Phase 2 rebuild the mix | **demoted** | The corpus already moves the metrics it was meant to fix. Filtering at 0.50 halves the mix, and less data on a model whose problem is competence is the wrong direction. The *measure* stays (it is built, AUC 0.98); the *rebuild* waits for a reason. |
| Phase 3 GRPO | **promoted to next** | The gap is "ambitious **and** renders". A render is a verifiable reward, which is precisely the tool for that gap, and nothing else on the list attacks it directly. |
| — | **new: cheap competence probes first** | Before spending 20 GPU-hours on RL, three one-variable runs that cost hours, not days. |

**Two probes ruled out before spending a GPU-hour on them**, by reading run
17's own `trainer_state.json`:

    eval_loss  0.6317  0.5836  0.5692  0.5604  0.5619  0.5611  0.5609
    step          60     120     180     240     300     360     377
    train loss 0.80 -------------------------------------------> 0.49

Eval loss stopped improving at step 240 of 377 and flattened while train
loss kept falling. **Run 17 was not undertrained; it was slightly
overtrained.** So:

- *Train longer / more epochs* — no. The model has extracted what this mix
  contains, and a second epoch buys overfitting.
- *Upweight gold* — no, for the same reason and worse. The positive set is
  44 unique scenes already repeated six times; tripling that overfits 44
  scenes harder. It was also a confounded probe: gold is the most ambitious
  code in the mix, so more of it raises ambition as well as correctness,
  which is the opposite of what needs testing.

Worth keeping from this: run 17 could have stopped at step 240 with the same
eval loss. Early stopping belongs in the kernel.

**Both scaffolding probes ran, and both failed.** 2026-09-22:

| probe | result |
|---|---|
| pre-render API check (`forge/repair/apicheck.py`) | catches **1 of 23** failures, 0 false positives on 77 passes |
| retrieval in the repair prompt (`--repair-retrieval`) | **77% vs 77%**, rescued 4 vs 4, one won one lost |

The second is the informative one. Same adapter, same prompts, one flag; the
first-round error mix is identical (73 none / 25 api_misuse both ways), so
generation is untouched and repair is the only variable — and it moved
nothing.

**The model is not failing for lack of examples.** It is committed to an
approach it cannot execute, and putting a correct scene in front of it
during repair does not make it abandon that approach. Nor is the failure a
name it invented: the API check proves only one of the 23 is a symbol that
does not exist.

That closes the cheap tier. The gap is not in the scaffolding, so it is in
the weights or in the shape of the task — which is what the planner split
and GRPO address, and why they are what is left.

**The probes that survive, in order of cost.**

1. **Retrieval during repair.** Retrieval is used at generation and not at
   repair — repair gets an API briefing built from the traceback. The
   failures are architectural (`Polyhedron` hand-built where
   `Dodecahedron()` exists), and an example of the right construct is
   exactly what a briefing does not supply. This is an inference-time
   change: no training, measurable in one eval.
2. **Repair rows in the mix.** Every training row is "prompt → complete
   scene"; none is "here is an error, fix it". Worth testing, but demoted
   from the original reading: repair collapsed *because the approach was
   wrong*, and teaching the model to repair small errors does not teach it
   not to attempt a hand-built dodecahedron. Requires instrumenting
   `forge/repair/loop.py` to keep each round's code and error, which no
   artefact on disk currently does.

Only then GRPO, with the render as the primary reward and the Phase 2
measure as a denser secondary term.

**What would overturn this.** The control run is still going. If it comes
back materially below the recorded 85.2% / 8.4% baseline, then part of what
looks like a run-17 effect is a difference between the recorded baseline and
today's machine, and these numbers need recomputing before anything is built
on them.


Each ends in a number that decides whether the next is worth doing. Nothing
runs in parallel with something it could confound.

### Phase 0 — Foundations · **done**

| | |
|---|---|
| Process groups killed properly, tested | `45c98d6` |
| Gold export reaches the dataset | `bffb1dc` |
| Generated ids identify their row | `bffb1dc` |
| Static rows out of training | `bffb1dc` |
| Ledgers distinguish killed from failed | `796fc4d` |
| Every artefact catalogued | `796fc4d` |
| Docs re-derived from data | `83ff933` |
| 43 tests | `90e20bb` |
| `add_manim_import`, +38 rows recovered | `6131bb1` |
| Lint recoveries reach the export | `6131bb1` |
| `python -m forge.doctor` — 9 invariants | `6131bb1` |

### Phase 1 — Train, finally · **~1 week**

The project's entire premise — that a render-gated corpus beats a scraped one
— has never been tested. This was blocked on "the corpus isn't large enough",
a threshold nobody defined and which the audit shows was measuring the wrong
thing anyway.

**Setup**
1. `kaggle/kernel-metadata.json` so the notebook can be pushed by API rather
   than by hand. This is what makes training automatable later.
2. Switch `kaggle/01_sft.py` to **Unsloth + QLoRA**. A 7B in 4-bit is ~5 GB
   against a T4's 15 GB, and Unsloth is roughly 2× faster at ~60% of the
   memory. Our mix is ~2.7M tokens per epoch: minutes, not hours.
3. Upload the dataset to HF (private) and mirror it as a Kaggle dataset.

**The run.** Three epochs, LoRA r=16, on the current 3,010-example mix.
**One variable** — nothing else changes. Not the repair budget, not
retrieval, not the system prompt, not the base model. That rule has been
broken three times already and each break cost a result.

**Evaluation.** Both existing benchmarks, unchanged, against the two
baselines already on record: 100-row single-scene (93%) and 81-task hard eval
(85% render, 1.76% length, 8.8% coverage).

**What makes this interesting is not the render rate.** It will probably rise
a few points, and that is the least important number. The question is whether
**length ratio and concept coverage move at all.** The corpus analysis
predicts they will not. If that prediction holds, Phase 1 is the experiment
that proves Phase 2 is the whole project — which is worth a week.

*Exit: two numbers against two baselines in `RESULTS.md`, with the hash of
the mix that produced them.*

### Phase 2 — A gate that judges animation · **measure done 2026-09-22**

The real work. Everything before it is preparation.

The render gate answers *did this produce a video file*. A second gate has to
answer *is this an explanation*. Structure first, not a model-judge — things
that can be measured, argued about, and cheated only by actually improving.
**The 44 gold scenes are the labelled positive set**, which is what makes a
judge unnecessary.

`forge/gate/animation.py` exists. It separates gold from the corpus at
composite **AUC 0.98**, stable across four disjoint samples (0.970–0.979),
with both halves of the gold set separating equally (0.978 / 0.967). About a
quarter of corpus rows still reach the weakest gold scene — the honest
ceiling of this version.

**Most of what this section originally proposed measuring does not work,
and finding that out was the point.** Per-signal AUC against 2,770 non-gold
rows:

| signal | AUC | |
|---|---|---|
| `computed_labels` | 0.926 | numbers the scene works out |
| `n_play` | 0.844 | |
| `declared_seconds` | 0.813 | |
| discrete per 100 calls | 0.226 | gold animates *less* per unit of work |
| `continuous` | 0.538 | **coin flip** |
| `transforms` | 0.462 | **coin flip** |
| `geom_objs` | 0.534 | **coin flip** |

- **Continuous constructs do not separate a gold scene from a corpus row.**
  The audit's headline — ValueTracker in 9.1% of gold against 1.3% of the
  corpus — is a *corpus-level ratio resting on four scenes*. The other 40
  gold scenes do not use ValueTracker either. This section was designed
  around that signal; it is noise per scene.
- **Text-heavy does not mean slideware.** Gold averages 16.4 text mobjects
  against 7.7. 3Blue1Brown scenes label things. The "ratio of text to
  geometric mobjects" bullet is deleted rather than inverted, because
  inverting a signal to match the labels is fitting.
- **`assert` separates perfectly (40/44 vs 0/2770) because it is a house
  convention**, not because asserted scenes animate better. Reported as
  `convention`, kept out of the score.

**What remains: rebuilding the mix, and the cut is expensive.**
`scripts/score_mix.py` scores all 3,010 rows:

| source | rows | mean |
|---|---|---|
| synthetic | 1308 | 0.454 |
| bespoke | 888 | 0.311 |
| thanhkt | 448 | 0.402 |
| gold | 240 | 0.727 |
| manimbench | 126 | ~0.23 |

A threshold of 0.40 keeps 1,469 rows; 0.50 keeps 754, a third of them gold;
0.70 keeps 221 and is essentially the gold set. Gold averages 0.727, so a
cut near it leaves too little to train on and a cut that preserves size
keeps most of what the measure calls slideware. **That trade is the
decision** — and the 888 bespoke rows scoring below teacher-generated ones
is its own question.

Only after that: rewrite the synthetic prompts to ask for what the measure
rewards. New generation restarts here and not before.

*Exit: a rebuilt mix whose profile sits between the corpus and gold, trained
once, measured against the Phase 1 numbers.*

### Phase 3 — Length, structure, and GRPO · **~2 weeks**

Two things Phase 2 does not address.

**Length.** A 16-minute explainer is not one scene. The `decompose` task type
was written for this and has never been evaluated: plan → several scenes →
render → concatenate, with the beat system as the unit. Deliberately after
Phase 2, because a chapter made of eight slideware scenes is worse than one.

**GRPO.** The render gate is a verifiable reward — exactly the RLVR setup
that GRPO is designed around, and the reason to have kept reward and
verification in the same package. The Phase 2 measure becomes a second,
denser reward term. Runs on Kaggle, where the gate can execute alongside
training; the 30h/week quota is the constraint, and a long run has to
checkpoint across 9-hour sessions.

*Exit: length ratio and coverage against the Phase 1 numbers.*

### Phase 4 — Serve it · **~1 week**

`docs/ARCHITECTURE.md` is written and costed against measured latency.
Nothing is built. Last on purpose: there is no point serving a model that
answers a smaller question than the one asked, and nothing in Phases 1–3
changes the architecture.

- Inference and render as **separate** Modal functions. They must not share a
  container: one wants a GPU for 13 seconds, the other wants CPU cores for
  minutes, and scaling them together wastes whichever is idle.
- Videos to Cloudflare R2. Egress is free there, which for a video product is
  the whole decision — the same bytes out of S3 would dominate the bill.
- Frontend on Cloudflare Pages. Static assets are unmetered.
- A queue, because a render is minutes: submit, poll, notify.

### Phase 5 — Autonomy · **ongoing, starts after Phase 1**

See §5. Built incrementally — every phase that works by hand gets automated
once, and not before.

---

## 4. Infrastructure — researched, costed, decided

Everything below was checked in September 2026. Free tiers move fast: four
providers (Cerebras, GitHub Models, Together, SambaNova) dropped or gated
their free LLM tiers between June and September alone, so the teacher layer
is built to rotate and to survive any one of them vanishing.

### Decided

| need | choice | cost | why |
|---|---|---|---|
| **SFT + GRPO training** | Kaggle notebooks | free | 30 h/week GPU, 9 h sessions, dual T4 (2×16 GB) or P100; root access so LaTeX and ffmpeg install and the render gate runs *in the same place as training* — which GRPO requires, since the reward is a render |
| **Training overflow** | Modal | $30/mo credit | ≈ 50 GPU-hours of T4, or 37 of L4. The escape hatch when Kaggle's quota is gone mid-experiment |
| **Model + dataset storage** | Hugging Face | free | 100 GB private, effectively unlimited public. Our data is 460 MB |
| **Inference serving** | Modal | within $30/mo | L4 at $0.000222/s; median generation 13.4 s ⇒ **$0.003/video**. Scale-to-zero, per-second billing, custom images |
| **Render serving** | Modal, separate function | within $30/mo | CPU at $0.0000131/core-s; a 1080p scene ≈ 120 core-s ⇒ **$0.0016/video** |
| **Video storage + CDN** | Cloudflare R2 | free to 10 GB | **zero egress**, which for video is the entire decision |
| **Frontend** | Cloudflare Pages | free | unlimited bandwidth on static assets; 100 k Worker requests/day |
| **Domain** | Cloudflare Registrar | ~$8.50–10.50/yr | at-cost — wholesale plus the ICANN fee, no markup, no renewal trap |
| **Scheduler / CI** | GitHub Actions | free **if the repo is public** | unlimited minutes and working `schedule:` on public repos. On a free *private* repo: 2,000 min/month and **cron is disabled** — see §5 |
| **Teacher models** | Gemini + Mistral, rotating | free tiers | ~900–1,000 Gemini calls/day *total*, measured, not the documented 1,500/model |

**Running cost of the whole product: roughly $0.005 per video**, against
Modal's $30/month credit — about **4,000–6,000 videos a month at zero
marginal cost**, plus ~$10/year for the domain. That is the entire budget.

### Rejected, and why

- **HF Spaces for inference** — free ZeroGPU is 3.5 minutes of GPU per *day*.
  Fine for a demo, unusable as the product. Free CPU Spaces sleep after 48 h.
- **Replicate** — fastest to a working endpoint, roughly 2× Modal per
  GPU-hour at scale, and no free credit.
- **Baseten** — more depth for custom models than Replicate, but H100-class
  pricing with no free tier.
- **RunPod serverless** — cheapest per-second on A100/H100 and sub-200 ms
  cold starts, but we are L4-class and spiky; Modal's free credit decides it.
- **S3 / GCS for video** — egress would be the dominant line item. R2 is free.
- **Supabase** — 500 MB Postgres free, but projects **pause after one week of
  inactivity**, which is disqualifying for something meant to run unattended.
- **A database at all, initially** — Modal's own Dict and Queue cover a job
  queue and a session log. Add Neon (scale-to-zero, no pausing) when there is
  a reason.
- **Qwen3-Coder-Next as the base model** — it is the current flagship small
  coder and it is 80B total / 3B active. Too large to fine-tune on any free
  tier. A dense 7B stays right. *Qwen3-Coder 7B* is a legitimate upgrade
  candidate, but changing the base model is its own experiment and must not
  ride along with Phase 1.
- **Cerebras / Together / SambaNova / GitHub Models as teachers** — all
  dropped or gated their free tiers this year. Kept in the provider rotation
  as optional, never depended on.

---

## 5. The autonomous loop

The goal: after the setup in `docs/SETUP.md`, the project advances without
anyone at a keyboard. Three tiers, because the three resources fail
differently.

### Tier 1 — GitHub Actions, the heartbeat

**This requires the repository to be public.** On a free private repo,
scheduled workflows do not run at all, and the choice between "pay $4/month
for Pro" and "publish the code" is not close for a project whose stated end
is free and educational. Only code goes public; `data/` stays gitignored and
lives on Hugging Face.

Scheduled jobs, all cheap and all CPU:

- **`doctor`** — nightly. Re-derives every figure, checks nine invariants,
  opens an issue if one breaks. This is the thing that would have caught four
  of the defects in the postmortem on the day they landed.
- **`tests`** — on every push.
- **`train`** — pushes the Kaggle notebook by API, polls `kernels status`,
  pulls the output, commits the eval numbers. This is the piece that makes
  training autonomous rather than a thing someone remembers to do.
- **`backup`** — corpus and renders to HF on a schedule.

### Tier 2 — the Mac, under launchd

CPU work that wants the local machine and no quota: showcase renders,
re-gating after a lint rule lands, the render side of any eval. Driven by
`scripts/forge_run.py`, which is now honest about what a job has produced —
but **not restarted until Phase 2**, because its job was to generate more
rows of the kind this plan says not to generate.

### Tier 3 — Modal, on demand

Inference and render for real users. Scale-to-zero, so it costs nothing when
nobody is asking.

### What keeps it honest

The audit's lesson, built in rather than remembered:

1. **Every producer names its consumer.** Four separate times, a step wrote a
   file nothing read. A `doctor` invariant now asserts arrival, not
   production, for gold scenes and lint recoveries; any new producer adds one.
2. **Health is progress, never existence** — already in `orchestrator.py`.
3. **A job that finishes with nothing to do stays down** and says so once.
4. **Budget guards.** Modal has a hard spend cap; the free credit is the
   budget and the cap is set at it. An autonomous system with a credit card
   attached and no ceiling is the one failure mode that costs real money.
5. **One variable per experiment**, enforced by recording the mix hash and
   config with every eval number.

---

## 6. Standing rules

- One variable per experiment. Broken three times; each break cost a result.
- Preserve every artefact, failures included. (`data/raw/` is empty — the
  3,753 rows the dedupe discarded were never kept, so its cost cannot be
  audited without re-downloading. Fix when ingestion next runs.)
- Busy is not the metric. Produced-and-arrived is the metric.
- Never commercial. CC BY-NC-SA 4.0.
- Do not ship something that merely renders — which is now also the
  definition of the Phase 2 gate.
