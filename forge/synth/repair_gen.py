"""Generate, gate, and repair — the teacher loop with the same cure we use locally.

The synth pipeline generated once and gated once, so every fixable failure was
thrown away. That is the single biggest waste in the data pipeline: `api_misuse`
accounts for roughly half of all failures, and it is precisely what the repair
loop fixes, because the repair prompt carries the *real* signature introspected
from the installed Manim rather than asking the model to guess again.

The same loop lifted the local model from 40% to 70% on the benchmark. Applying
it here costs one extra call per failure and recovers rows we already paid to
generate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from forge.harness import RenderHarness, RenderResult, is_repairable
from forge.harness.errors import tail
from forge.repair.api import api_briefing
from forge.repair.lint import lint
from forge.synth.teacher import Teacher, extract_code


ESCALATION = (
    "You have now failed twice with the same kind of error, so the approach "
    "itself is wrong, not the details.\n"
    "Do NOT try to fix the failing call. DELETE it and achieve the same visual "
    "effect using only the simplest, most common Manim objects and animations. "
    "A simpler scene that renders is worth far more than an elaborate one that "
    "does not."
)


def repair_instruction(original_request: str, code: str, result: RenderResult,
                       repeated: bool = False) -> str:
    parts = [
        "The scene below was written for this request:\n",
        original_request.strip(),
        "\n\nIt failed to render.\n\n```python\n" + code + "\n```\n",
        f"Error ({result.error_kind.value}):\n```\n{tail(result.stderr, 10)}\n```\n",
    ]
    briefing = api_briefing(code, result.stderr)
    if briefing:
        parts.append(briefing + "\n")
    if repeated:
        parts.append(ESCALATION + "\n")
    parts.append(
        "Rewrite the complete scene so it renders, keeping the beat structure, "
        "the narration= arguments and the computed values. "
        "Output only code in one ``` block."
    )
    return "\n".join(parts)


@dataclass
class RepairStep:
    """One failed attempt and what was wrong with it.

    Kept in full because (broken code -> error -> fixed code) is the rarest
    and most useful training data this pipeline produces: it is what teaches a
    model to read its own traceback instead of guessing again. Recording only
    the final code throws it away, which is what an earlier version did while
    generating hundreds of them.
    """
    code: str
    error_kind: str
    stderr_tail: str


@dataclass
class GenResult:
    ok: bool
    code: str
    rounds: int
    error_kind: str
    history: list[str] = field(default_factory=list)
    lint_rules: list[str] = field(default_factory=list)
    duration_s: float | None = None
    n_frames: int = 0
    #: Every failed attempt, oldest first. Paired with ``code`` when ``ok``,
    #: these form the repair training set.
    attempts: list[RepairStep] = field(default_factory=list)


def generate_and_repair(teacher: Teacher, harness: RenderHarness, request: str,
                        n_beats: int, length_hint: str,
                        max_rounds: int = 4,
                        first_code: str | None = None) -> GenResult:
    """``first_code`` lets a caller supply a generation it already made, so a
    pipeline that probes before committing does not pay for the same call twice."""
    code = first_code if first_code is not None else teacher.generate(
        request, n_beats, length_hint)
    code, rules = lint(code)
    result = harness.render(code, quality="low", frames=4)
    history = [result.error_kind.value]
    attempts: list[RepairStep] = []

    rounds = 0
    while not result.ok and rounds < max_rounds:
        # An environment failure indicts the machine; rewriting cannot fix it
        # and would burn a generation against the free-tier quota.
        if result.is_environment_failure or not is_repairable(result.error_kind):
            break
        rounds += 1
        attempts.append(RepairStep(code=code, error_kind=result.error_kind.value,
                                   stderr_tail=tail(result.stderr, 10)))
        repeated = len(history) >= 2 and history[-1] == history[-2]
        code = extract_code(teacher._gemini_rest(
            repair_instruction(request, code, result, repeated), 16000)
            if teacher.native else
            teacher.generate(repair_instruction(request, code, result, repeated),
                             n_beats, length_hint))
        code, more = lint(code)
        rules += more
        result = harness.render(code, quality="low", frames=4)
        history.append(result.error_kind.value)

    return GenResult(
        ok=result.ok, code=code, rounds=rounds,
        error_kind=result.error_kind.value, history=history,
        lint_rules=rules, duration_s=result.duration_s,
        n_frames=len(result.frame_paths), attempts=attempts,
    )
