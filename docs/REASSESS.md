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
