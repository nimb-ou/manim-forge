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
