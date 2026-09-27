# The six-hour reassessment

Run this every six hours. It exists because on 2026-09-23 I spent most of a
day on a watchdog's scheduling — four process-manager changes — when the
fault was `time.sleep(300)` not returning, and nobody stopped me. Nimit asked
for the check after watching that happen.

The point is not to feel bad about a detour. It is to notice one **while it
is still cheap**, and to ask whether the plan still matches what has been
learned since it was written.

## 1. What has arrived since the last check?

Not what is running. What is **on disk and verified** — an adapter, a
measured number, a row count that grew. If the answer is "a lot of fixes and
no result", that is the signal.

    git log --oneline --since="6 hours ago" | wc -l
    ls -lt data/bench/ | head -5
    tail -20 data/supervisor/supervisor.log

## 2. Am I fixing the thing, or fixing where the symptom appeared?

The day's pattern, six times over: a failure in the harness reported as a
failure of the model. And once — the sleep — four fixes aimed at the
scheduler because the symptom was a missed schedule.

- How many attempts has the current problem had?
- **Three or more without progress: stop and test the assumption under the
  fix**, not a different fix.
- Is there a cheaper measurement that would tell me which half is wrong?

## 3. Is this on the critical path?

The critical path is: *does a render-gated corpus produce better
explanations than an ungated one, and can a 7B execute them.* Everything
else is support.

- Would the thing I am doing change a number in `docs/RESULTS.md`?
- If it broke entirely, what would actually stop?
- Infrastructure earns its place by unblocking a measurement. If the last
  three commits are infrastructure, the next one should not be.

## 4. What is the plan missing?

- Which assumption in `docs/PLAN.md` has been contradicted since it was
  written and not yet edited? (§2's prediction was wrong; §2b's
  ValueTracker framing was wrong; both were corrected late.)
- What am I measuring that I cannot yet trust, and what would make it
  trustworthy?
- What has been sitting unused? The 151 narration arcs sat unused since the
  scrape and turned out to be the planner's whole training set.

## 5. What would I tell someone taking over?

If the honest answer is "I am three fixes deep into something I cannot
explain the value of", write that down, stop, and go back to §3.

---

## Log · 2026-09-23 23:20Z

**1. Arrived (verified, on disk).** Coder v3 adapter (92.7% beat eval — same
as v2). Planner v2 adapter (loops: ~45 repeats per 48-beat plan, no END).
Beat-level eval: the coder alone is not the bottleneck. Kaggle renders
Manim + LaTeX (2.6 s, four in parallel 5.1 s) — GRPO's reward is feasible.
Split end-to-end 3/8 → **5/8** (planner v2 + coder v2). Data: 1,805 arcs,
1,513 topics, ~1,000 short-request variants.

**2. Symptom or thing?** Four harness faults found and fixed today, each
with a measured before/after; the step to 5/8 is those, not a model. That
is the recurring pattern, and it is still paying — but it also means every
number from before today's fixes understates the models. No problem has had
three attempts without progress. The one candidate: *coder data cleaning*
(v3) produced no gain on either measure, so a coder v4 by more cleaning
would be a third attempt. Not doing it.

**3. Critical path.** The 24-title hard eval against run 17 is running now
— that is the number the plan is waiting for. Last three commits: two
measurements, one fix. Fine.

**4. What the plan is missing.**
- Plans are now 4.8 beats (stopped at the first repeat) against ~40 in real
  arcs. Coverage on the hard eval will be capped by that. Planner v3 (clean
  arcs) is the fix; *if v3 still loops, test sampling (temp 0.3–0.5,
  repetition penalty) before training again* — a free test.
- Coder SFT is done. The coder's remaining losses are composition, so GRPO
  should reward **whole assembled scenes, with the coder conditioned on its
  own earlier beats**, not single beats on reference prefixes (92.7%
  already, little headroom).
- 1,805 arcs is 7× what planner v2 had. Planner v4 on all of it, cleaned,
  after v3 says whether cleaning worked.

**5. Handover.** Best pair: planner v2 + coder v2, 5/8. Demo:
`scripts/demo.py`. Next decisions wait on planner v3 (training) and the
hard eval (running).

## Log · 2026-09-24 05:25Z

**1. Arrived.** Sampled planning (22.6 beats before the first repeat, from
3.8 greedy) — the biggest single gain of the night, and it cost no
training. Runtime salvage. Planner v3 adapter on disk. The first 24-title
hard eval (13/24 render, coverage 4.5%). Arc referee + fixer pipeline.
Kaggle GPU quota spent at 02:28Z.

**2. Symptom or thing?** The last six hours are mostly infrastructure: a
readiness gate (after the supervisor collected v3 as v4), an alarm that
killed its own job, a judge calibrated and one thrown out (magistral passed
"1/2 · 1/4 = 1/2"). Each was a real fault, but that is the "lots of fixes"
signal. **The next work is measurement, not plumbing.**

**3. Critical path.** Planner v3 has been on disk ~3.5 h unevaluated
because the Mac is running the sampled hard eval, which is also on the
critical path. The chain runs v3 immediately after. The data-quality
pipeline feeds planner v4, which cannot train before the quota resets, so
it is background — it gets no more of my time unless it breaks.

**4. Missing.**
- Sampling makes every number noisier. n=8 comparisons are now close to
  meaningless; decisions need the 24-title runs or repeats.
- GRPO has no design beyond "whole-scene reward". With the GPU gone for two
  days, write it now so it can run the hour the quota returns.
- The demo is what Nimit will try first. Run it once, sampled and with
  runtime salvage, before morning.

**5. Handover.** Best pair planner v2 (sampled) + coder v2. Hard eval
running; then planner v3 evals; planner v4 queued for the quota reset with
referee-checked arcs.

## Log · 2026-09-24 11:25Z

**1. Arrived.** Planner v3 measured (33.9 beats before a repeat; loops by
counting, now caught). The hard eval with planner v3 + all fixes: 10/21
rendered so far with 11–25-beat scenes, against 13/24 of 3.8-beat scenes
last night. GRPO kernel + 195 prompts + reward checked locally. Referee:
988 arcs judged, 705 corrected, only 121 of 432 corrections pass re-check.

**2. Symptom or thing?** Nine commits: two measurements, five harness
fixes that each came *from* a measurement (lambda args, counting loops,
truncated beats, set-up repair), one refactor, one supervisor fix. Better
ratio than the last window. The harness keeps yielding real faults because
longer scenes exercise more of it — that is expected, but the rule stands:
each fix needs a before/after number, and the truncation fix does not have
one yet.

**3. Critical path.** The hard eval result (≈20 min) is the go/no-go on the
split against run 17. Next Mac job: the same 24 titles with the truncation
fix, so that fix gets its number.

**4. Missing.**
- The fixer's 28% pass rate means correcting arcs mostly fails; planner v4
  will lean on fewer, correct arcs. Do not build more on the fixer — let it
  finish and use what passes.
- **Max tokens per beat (900)** is producing truncations on long beats. A
  larger cap is a one-flag test; do it with the rerun instead of more
  salvage logic.
- Nothing has been rendered at medium quality and *looked at* since the
  demo. A render passing is not a scene being good. Look at two of today's
  rendered hard titles.

**5. Handover.** Best: planner v3 (sampled, count-aware repeat stop) +
coder v2 + salvage + set-up. Kaggle quota out; v4 and GRPO queued.

## Log · 2026-09-24 17:30Z

**1. Arrived.** Nimit watched the best hard-eval render and called it
"complete nonsense" — correctly. It is text slides; 215 of 289 beats built
no picture. Everything measured this week (renders, coverage, length) was
blind to that.

**2. Symptom or thing?** The *thing*: a 7B model writing raw Manim falls
back on Text because it is the one call that always works. Four days of
salvage, repair and data cleaning improved whether scenes *render*, not
whether they *show anything*. That is the rabbit hole: optimising the
measurable proxy. More harness work on raw Manim would be a fifth layer of
the same thing.

**3. Critical path — changed.** New centre: **the Forge kit** (forge/kit),
~35 3Blue1Brown-style blocks (planes, vectors, matrix moves,
eigenvectors, tangents, Riemann sums, Taylor, unit-circle sine, complex
multiplication, neural nets, sampling histograms) behind a Stage that owns
layout. The model chooses blocks and parameters; the picture is
guaranteed by construction. Next numbers: kit vs raw on 12 hard titles
(queued, with a smoke first), judged by *visual-beat share* and contact
sheets looked at, not by coverage.

**4. Missing.**
- The coder has never been trained on the kit; the A/B is prompt-only. If
  it helps, the data for coder v5 is teacher-written kit beats, render-
  verified and visual-checked — build that generator now.
- GRPO, planner v4 and coder v4 were designed around raw Manim; coder v4
  (gate-filtered raw data) is now likely the wrong next GPU job. Re-order
  when the quota resets: kit-trained coder first.
- Phase 4 app built (forge/serve), kit on by default — ready for Nimit to
  use once the Mac is free.

**5. Handover.** The product question is now "does the kit make pictures
that explain", answered by looking at them.

## Log · 2026-09-24 23:40Z

**1. Arrived.** The kit: ~50 blocks, four galleries rendered and looked at.
Web app (forge/serve). Kit teacher data: ~2,000 raw rows, about half
surviving the relevance and novelty filter. Measured: prompt-only kit mode
keeps 0 of 4 beats (the untrained coder copies the example) — so the kit
must be trained in. Kit coder v5 training locally now (first attempt went
NaN on two over-long rows; fixed).

**2. Symptom or thing?** The thing, this time: the pictures. Each
teacher-data problem was found by *looking* — Escher as ten vector
diagrams, stage.play errors — and fixed at the source (relevance filter,
forgiving Stage, verb names), not by salvage logic.

**3. Critical path.** Kit coder v5 → 12-title scorecard judged on visual
share and contact sheets. Nothing else is on it. The planner and GRPO wait
for the Kaggle reset.

**4. Missing.**
- Local training is slow (~60 s/iteration under contention; shards paused
  to help). If v5 takes more than ~10 h, train a smaller first cut
  (fewer rows) to get *a* number, rather than waiting for a perfect one.
- The relevance filter is keyword-based; it will pass some wrong pictures.
  The scorecard's contact sheets are the check.
- The planner still writes intents for a raw-Manim coder. Once v5 works,
  the planner's intents should name drawable things; that is planner v5.

**5. Handover.** If v5 draws the right pictures, the product is the web
app with the kit coder; if not, the next lever is more, better kit data
(the teacher is the ceiling), not more harness.

## Log · 2026-09-26 22:35Z

**1. Arrived.** Kit v5 58% visual-and-new on the 20 short prompts (raw v2
33%). GRPO v1 gamed its reward (47%); GRPO v2, with the fixed reward, ran
332/400 steps, reward 0.5 → ~0.9 — being scored now, by eye as well.
Self-training: 164 arcs, 700/1,150 beats visual; 319 rows survive the
filters and are in kit v6 (SFT running on Kaggle). Mistral out until the
budget resets; Gemini critic rate-limited.

**2. Symptom or thing?** The thing. Kit v5's all-caption scenes were not
the model failing to draw: beat 1 drew, one call in it raised, the salvage
dropped the whole beat and the rest lost their axes. A fuzz of the kit
itself found nine blocks that fail on reasonable calls — shade_area and
riemann_refine on *every* plain function, slide_tangent on any falling
start. The model was being punished for the kit's bugs, in training data
(scenes that failed never became rows) and at inference.

**3. Critical path.** Same-kit comparison of GRPO v2, kit v5 and kit v5 +
planner v4 (queued, ~1.5 h), then kit v6 when it lands. Pick the best
coder/planner pair for the app.

**4. Missing.** The eval notes now carry the exception text of every
runtime drop: read them after this round and fix what repeats (kit or
prompt), before generating more data.

**5. Handover.** If the fixed kit moves kit v5 past 58% on its own, the
biggest lever was the kit, and the fuzz belongs in CI.

*22:40Z, looked:* kit v5's "integration as thinner rectangles" contact sheet
shows axes and x² in all six beats and not one rectangle — the
riemann_refine-on-a-lambda crash, dropped each time. Consistent with the
fuzz finding; the same prompt under the fixed kit is the first thing to
check in c5_fix. No plan change.

## Log · 2026-09-27 01:30Z

**1. Arrived.** Relevance (the training filter's subject test) is now in
the scorecard, over planned beats. Short prompts: raw 24%, kit v5 26% →
34% on the fuzz-fixed kit → 43% with subject hints + resampling; GRPO v2
34% → 44% with them. GRPO v2's 76% "visual" was a plane and a vector for
everything; caught by looking, again. Kit v6 collected (score queued).
GRPO v3 running on Kaggle: relevance reward, hints in prompts, from GRPO v2.

**2. Spiralling?** No, but note the pattern: twice now a GRPO metric was
gamed and only the contact sheets showed it. Every metric I add closes one
hole; the sheets stay the judge. Most of today's gain came from the kit
(bugs, tolerance) and inference (hints), not from training.

**3. Critical path.** Kit v6 vs GRPO v2 (both with hints) → GRPO v3 →
the best coder into the app with relevance on by default.

**4. Missing.** Looked at gradient descent (kit v5 + hints): on subject,
but it draws small arrows instead of gradient_descent's stepping ball, and
the parabola ran off the axes through the title (plot_graph now clips to
the axes' range). The model under-uses the motion blocks it knows —
the planner's intents rarely ask for motion. A planner that writes
"the ball steps downhill" would pull the right block; planner v4 vs v3
is queued.

**5. Plan change.** Relevance on by default in the app (it costs up to
3 extra samples per off-subject beat). Fuzz the kit whenever it changes.

## Log · 2026-09-27 08:20Z

**1. Arrived.** Six classic-picture blocks, 31 hand-written scenes, kit v7
(hand-written rows ×4, name hints in prompts), GRPO v3 (relevance reward).
Same kit, short prompts, relevance on: kit v6 49%, kit v7 41%; GRPO v3
scoring. Kit v7 is worse: it reaches for the new blocks and guesses their
arguments (swing_pendulum(stage, p, m, angle=30), superpose_waves(stage,
ax, [g1, g2])); two rounds of kit tolerance did not close the gap.

**2. Spiralling?** Partly. The last three rounds were kit tolerance for a
model's guesses — useful, but it chases each new guess. The root cause is
that the prompt names blocks without saying how to call them. Now the hint
carries each block's signature and one-line purpose (from the functions
themselves); that is one change, not another round of patches.

**3. Looked:** kit v6's "integration" scored 6/6 relevant-visual; the
sheet shows shading, then empty axes, a shaded block off the chart, one
tiny rectangle. The metric (any non-scaffold picture, on subject by
keywords) is lenient — real quality is well below 49%. Shading is now
clamped to the axes. A vision judge is the missing measurement: Gemini is
rate-limited; I can read the 20 sheets myself for the decisions that matter
(which coder ships), and should.

**4. Critical path.** Signature hints on kit v6 and GRPO v3 → pick the
coder by metric *and* by reading all 20 sheets → app. Kit v7 is shelved.

**5. Plan change.** Stop adding tolerance for single guesses; fix the
prompt (signatures). Next GPU week: SFT with signature hints in every
prompt, and GRPO from the best of those.

## Log · 2026-09-27 13:40Z

**1. Arrived.** Signature hints: kit v6 63%, GRPO v3 61% (tie; kit v6 in
the app). Self-training now generates with hints (1,374 rows). GPU quota
spent until ~Oct 3; Mistral back ~Sep 30.

**2. Looked — all 20 sheets of the shipped configuration (kit v6 + hints),
by eye.** Clearly good: 3 (dot product as projection, neural network
layers, binary counting). Mostly good: 2 (matrix on the plane, halving
series). Partial: 8. Poor: 7 (derivative, Taylor, multiply by i, Bayes,
gradient descent, primes — a binary counter!, pendulum). Eye score ≈ 43%
(good 1, partial ½) against the metric's 63%. The metric counts a bare
square or a lone arrow; the eye asks whether the idea is shown.

**3. Am I spiralling?** The metric has been climbing faster than the
pictures. Kit tolerance and hints were real gains (the sheets are better
than this morning's), but I have been optimising a lenient proxy on the
same 20 prompts that the kit's newest blocks and my hand-written scenes
were built around. Two corrections:
  * a **held-out prompt set** on topics no block or hand-written scene was
    made for (forge/evaluate/heldout_prompts.json) — the generalisation
    number;
  * **eye scores** on every run that decides what ships (20 sheets, good /
    partial / poor), recorded next to the metric.

**4. What the poor scenes share.** Beats that are only captions over
empty axes even where the right block exists (gradient descent never
calls gradient_descent; Taylor draws axes and one curve). The coder is
not choosing the block the beat needs. The lever I have not tried: show
it a worked scene for a *similar* request (retrieval of one hand-written
scene as an example). Untrained, the model copied examples verbatim; the
trained coder copying a gradient-descent example into a gradient-descent
request would be the right behaviour. Test on held-out prompts only, so
the example library cannot inflate the number.

**5. Plan.** (a) held-out prompts + eye-score both configurations;
(b) exemplar retrieval as an option, evaluated on held-out; (c) Oct 3:
SFT with signature hints + exemplars in prompts, then GRPO.

*17:20Z — the numbers were noisier than the differences.* Same
configuration (kit v6 + signature hints), held-out prompts: 56% unseeded,
37% with the planner seeded. The planner samples at temperature 0.5, so
each run scores the coder on different plans; on 20 prompts that moves the
score ~20 points. Today's comparisons (49 → 63% for signatures, exemplar
34 vs 56%) were single unseeded runs and are not established. Now: plans
cached per prompt (scorecard --plans) so configurations differ only in the
coder; an ablation on 40 prompts (short + held-out) is running — signature
hints, none, + exemplar, names only. Decisions wait for it; the app keeps
relevance + signatures (no evidence it hurts) and no exemplar.

## Log · 2026-09-28 19:20Z

**1. Arrived.** Ablation on identical cached plans (kit v6), half done.
Short prompts: signature hints 45% on-subject vs no hints 37% (visual-new
51% vs 56%: the hint trades generic pictures for on-subject ones).
Held-out, signature hints: 37%; held-out without hints, + exemplar and
names-only still running (~4 h). Self-training at 1,433 rows.

**2. Looked.** Bayes, same plan, both configurations (beat-end frames):
with hints, beats 1-2 are a real Bayes square, P(sick|+) = 0.09; without,
grids, overlapping labels and a formula. The metric and the sheets agree
on this pair. Both degrade after beat 2: overlapping Text labels, empty
axes, a formula alone. Coherence past the opening beats is the common
failure, with or without hints.

**3. Spiralling?** No — this round fixed the measurement (plans fixed,
beat-end frames, held-out set) instead of moving a number. The honest
figures are lower than yesterday's headline: ~40% of planned beats show
an on-subject picture, and by eye fewer.

**4. Critical path.** Finish the ablation → ship the best inference
configuration. Then the two levers left on this week's hardware:
  * the planner: intents like "Table of true positives" invite text
    beats; planner v4 vs v3 needs its own evaluation (plans differ by
    design, so several seeds each);
  * the teacher returns ~Sep 30: kit data written with signature hints and
    the new blocks, then SFT on Oct 3 when the GPU quota resets.

**5. Plan change.** None until the ablation ends; no new kit patches for
single guesses.
