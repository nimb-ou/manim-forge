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
