"""Generate -> render -> repair.

The published state of the art on this task found that inference-time
render-and-retry produced larger gains than the fine-tuning did. That is a
strong claim, and it is testable here because the render gate we built for the
corpus is exactly the signal the loop needs.

Each round is cheap-to-expensive:

1. **Lint** — deterministic fixes, no model call. Catches the whole
   ``empty_render`` class for free.
2. **Render** — the ground truth. Nothing else counts as success.
3. **Repair** — only if the failure is the model's fault. Environment failures
   are never sent to a model, because no amount of rewriting fixes missing TeX.

The repair prompt carries the traceback tail plus real signatures introspected
from the installed Manim, so the model is corrected with fact rather than
asked to guess again.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from forge.harness import RenderHarness, RenderResult, ErrorKind, is_repairable
from forge.harness.errors import tail
from .api import api_briefing
from .lint import lint

SYSTEM = (
    "You are an expert Manim Community Edition v0.19+ developer. "
    "Given a description of an animation, write complete, runnable Python code "
    "using `from manim import *` and a single Scene subclass. "
    "Always animate with self.play(...) and end with self.wait(). "
    "Output only code in one ``` block. No explanation."
)

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    blocks = _FENCE.findall(text)
    return (max(blocks, key=len) if blocks else text).strip()


def repair_prompt(description: str, code: str, result: RenderResult) -> str:
    """The retry message: what was asked, what was written, what broke, what's true."""
    parts = [
        f"This Manim scene was meant to do the following:\n{description}\n",
        f"The code below failed to render.\n\n```python\n{code}\n```\n",
        f"Error ({result.error_kind.value}):\n```\n{tail(result.stderr, 10)}\n```\n",
    ]
    briefing = api_briefing(code, result.stderr)
    if briefing:
        parts.append(briefing + "\n")
    parts.append("Rewrite the complete scene so it renders. Output only code in one ``` block.")
    return "\n".join(parts)


@dataclass
class LoopResult:
    ok: bool
    code: str
    rounds_used: int
    final_error: str
    lint_rules: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)   # error_kind per round


class RepairLoop:
    def __init__(self, model, tokenizer, harness: RenderHarness,
                 max_rounds: int = 2, max_tokens: int = 900):
        self.model, self.tok, self.harness = model, tokenizer, harness
        self.max_rounds, self.max_tokens = max_rounds, max_tokens

    def _generate(self, user: str) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        chat = self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False,
        )
        return generate(self.model, self.tok, prompt=chat,
                        max_tokens=self.max_tokens,
                        sampler=make_sampler(temp=0.0), verbose=False)

    def run(self, description: str) -> LoopResult:
        code = extract_code(self._generate(description))
        code, rules = lint(code)
        result = self.harness.render(code, quality="low", frames=4)
        history = [result.error_kind.value]

        rounds = 0
        while not result.ok and rounds < self.max_rounds:
            # An environment failure is the machine's fault; rewriting the
            # scene cannot fix it and would waste a generation.
            if result.is_environment_failure or not is_repairable(result.error_kind):
                break
            rounds += 1
            code = extract_code(self._generate(repair_prompt(description, code, result)))
            code, more = lint(code)
            rules += more
            result = self.harness.render(code, quality="low", frames=4)
            history.append(result.error_kind.value)

        return LoopResult(
            ok=result.ok, code=code, rounds_used=rounds,
            final_error=result.error_kind.value,
            lint_rules=rules, history=history,
        )
