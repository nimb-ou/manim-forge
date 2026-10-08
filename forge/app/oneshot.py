"""One shot: the whole scene in one call, adapted from the nearest hand-written ones.

The two-stage pipeline writes each beat in a separate call that sees the
earlier beats only as intents and variable names. Its failures, on the
sheets (docs/PLAN.md, 2026-10-07), are failures of that separation: by beat
4 the picture has nothing to do with beat 1, titles name what the planner
said rather than what is drawn, and a 7B planner cannot keep a story
consistent. Adding ~400 teacher scenes to the per-beat coder moved the
headline judge by one point.

Here one call writes every beat, so each beat is written looking at the
code of the ones before it, and the call is shown the two hand-written
scenes nearest the request (forge/kit/library.py) to adapt. The output is
the construct() body itself:

    # beat 1: <what the picture shows>
    # say: <narration>
    <statements>
    # beat 2: ...

which is also the format the library renders its scenes in, so a model
trained on these rows learns one format end to end. Assembly, salvage and
rendering are the two-stage pipeline's (forge.app.pipeline.finish).
"""
from __future__ import annotations

import re
import time

from forge.app.twostage import Beat, extract_code

ONESHOT_SYSTEM = (
    "You write a complete short 3Blue1Brown-style animation with the Forge "
    "kit, answering the request in 3 to 6 beats. Write only the body of "
    "construct(): for each beat a line `# beat N: <what the picture shows>`, "
    "a line `# say: <the narration>`, then the beat's statements. "
    "`stage = Stage(self)` already exists. The beats are one scene: build "
    "objects once, then move, label, transform and reuse them in later "
    "beats; call stage.clear() only when the picture really changes. Use the "
    "request's own numbers and make every number on screen correct. No "
    "class, no def, no imports. Pictures first; text only through "
    "stage.title, stage.caption, stage.label and stage.equation."
)

#: The one-shot engine's model (2026-10-08). Graded by eye on the in-scope
#: set, untuned with retrieval and the kit reference, same engine: Qwen3.5-9B
#: 13 good of 20, Qwen2.5-Coder-7B 10, the 7B fine-tuned for this task 6; on
#: held-out topics 8 against 3 for the fine-tune (docs/RESULTS.md).
BASE_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"

BEAT_LINE = re.compile(r"^\s*#\s*beat\s*(\d+)\s*[:.-]\s*(.*)$", re.I)
SAY_LINE = re.compile(r"^\s*#\s*say\s*:\s*(.*)$", re.I)


def system_prompt(api: bool) -> str:
    """With ``api`` the kit's reference is appended: an untuned model has
    never seen the kit and has only the examples to go on."""
    if not api:
        return ONESHOT_SYSTEM
    from forge.app.pipeline import _kit_api
    return ONESHOT_SYSTEM + "\n\nTHE KIT\n" + _kit_api()


def user_prompt(request: str, k: int = 2, exclude: set[str] | None = None,
                examples: list[dict] | None = None) -> str:
    from forge.kit.library import as_body, similar
    if examples is None:
        examples = [s for _, s in similar(request, k, exclude=exclude)]
    parts = ["SIMILAR SCENES (hand-written and checked; adapt what fits)"]
    for s in examples:
        parts.append(f"REQUEST: {s['request']}\n{as_body(s)}")
    parts.append(f"NOW WRITE THE SCENE FOR\nREQUEST: {request}")
    return "\n\n".join(parts)


def parse_scene(text: str) -> tuple[list[Beat], list[str]]:
    """Beats and bodies out of a one-shot reply. Lines before the first
    ``# beat`` marker (a stray title call, say) join the first beat."""
    code = extract_code(text)
    beats: list[Beat] = []
    bodies: list[str] = []
    lead: list[str] = []
    cur: list[str] | None = None
    for line in code.splitlines():
        m = BEAT_LINE.match(line)
        if m:
            if cur is not None:
                bodies.append("\n".join(cur).strip("\n"))
            beats.append(Beat(len(beats) + 1, None, m.group(2).strip()))
            cur = list(lead) if not beats[:-1] else []
            lead = []
            continue
        s = SAY_LINE.match(line)
        if s and beats and cur is not None and not beats[-1].narration:
            beats[-1].narration = s.group(1).strip()
            continue
        (cur if cur is not None else lead).append(line)
    if cur is not None:
        bodies.append("\n".join(cur).strip("\n"))
    return beats, bodies


def score(res) -> tuple:
    """How good a rendered sample looks without a judge: it rendered, kept
    every beat, shows no arithmetic slip (forge/app/checks.py), has few
    layout problems (kit.layout_issues), and has 3-6 beats. Higher is
    better."""
    from forge.app.checks import arithmetic_errors
    kept = sum(1 for b in res.bodies if b.strip())
    slips = len(arithmetic_errors("\n".join(res.bodies)))
    return (res.ok, kept / max(1, len(res.beats)), -slips,
            -sum(res.layout), 3 <= len(res.beats) <= 6)


def run_oneshot(request: str, model, tok, emit=lambda e: None, opts=None,
                harness=None, api: bool = False, k: int = 2,
                max_tokens: int = 2400, exclude: set[str] | None = None,
                samples: int = 1, temp: float = 0.7):
    """Generate, parse, assemble and render as the pipeline does.

    With ``samples`` > 1 the first sample is greedy and the rest are drawn
    at ``temp``; each is rendered and the best by ``score`` is returned.
    A sample that renders whole with no layout problem ends the search.
    """
    from forge.app.pipeline import Options, Result, ask, finish
    opts = opts or Options(kit=True)
    t0 = time.time()
    system = system_prompt(api)
    user = user_prompt(request, k, exclude=exclude)
    best = None
    for n in range(samples):
        emit({"stage": "plan", "status": "start",
              **({"note": f"sample {n + 1} of {samples}"} if samples > 1 else {})})
        reply = ask(model, tok, system, user, max_tokens=max_tokens,
                    temp=temp if n else 0.0)
        beats, bodies = parse_scene(reply)
        emit({"stage": "plan", "status": "done", "n": len(beats),
              "elapsed": round(time.time() - t0, 1)})
        if not beats:
            res = Result(request, [], [], code=reply, error="no beats")
        else:
            for b, body in zip(beats, bodies):
                emit({"stage": "code", "beat": b.n, "intent": b.intent,
                      "code": body})
            res = finish(request, beats, bodies, model, tok, system, opts,
                         emit if samples == 1 else (lambda e: None),
                         harness, t0, [])
        if samples > 1:
            res.notes.append(f"sample {n + 1}/{samples}: score {score(res)}")
        if best is None or score(res) > score(best):
            best = res
        if best.ok and score(best)[1:4] == (1, 0, 0):
            break
    if samples > 1:
        emit({"stage": "done", "ok": best.ok, "video": best.video,
              "duration": best.duration, "error": best.error,
              "code": best.code, "elapsed": round(time.time() - t0, 1)})
    if not best.beats:
        emit({"stage": "done", "ok": False, "error": "no beats in the reply"})
    best.seconds = time.time() - t0
    return best
