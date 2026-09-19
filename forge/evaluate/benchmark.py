"""Baseline benchmark — how well does an untuned model write Manim?

Fine-tuning only earns its place if it beats the model we started with, so we
need that number before training anything. This measures it on ManimBench's
held-out 100-row test split, which the corpus pipeline never touches.

The headline metric is **Render Success Rate**: the fraction of generated
scenes that actually execute and produce video. It is deliberately crude and
deliberately honest — a scene that does not render is worth nothing regardless
of how plausible its code looks, and RSR is the one number that cannot be
argued with.

Environment failures are excluded from the denominator. If TeX breaks midway
through a run, those rows say nothing about the model and must not be counted
against it.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from forge.harness import RenderHarness, ErrorKind

SYSTEM = (
    "You are an expert Manim Community Edition v0.19+ developer. "
    "Given a description of an animation, write complete, runnable Python code "
    "using `from manim import *` and a single Scene subclass. "
    "Output only code in one ``` block. No explanation."
)

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    """Pull the code out of a model response.

    Models wrap code in fences inconsistently and sometimes omit them entirely,
    so fall back to the raw text when no fence is present rather than scoring a
    valid answer as a failure of formatting.
    """
    blocks = _FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


@dataclass
class Trial:
    index: int
    prompt: str
    ok: bool
    error_kind: str
    gen_seconds: float
    render_seconds: float
    code_chars: int
    n_play_calls: int          # animated, or merely static? see the ManimBench finding
    code: str = ""


class Benchmark:
    def __init__(self, harness: RenderHarness, max_tokens: int = 900):
        self.harness = harness
        self.max_tokens = max_tokens

    def run_model(self, model_id: str, prompts: list[str], out: Path) -> dict:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler

        print(f"loading {model_id} ...", flush=True)
        t0 = time.monotonic()
        model, tokenizer = load(model_id)
        print(f"  loaded in {time.monotonic()-t0:.1f}s", flush=True)

        # Greedy: we are measuring the model, not sampling luck, and the run
        # must be reproducible when we re-measure after fine-tuning.
        sampler = make_sampler(temp=0.0)

        trials: list[Trial] = []
        for i, prompt in enumerate(prompts):
            chat = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False,
            )
            t = time.monotonic()
            raw = generate(model, tokenizer, prompt=chat,
                           max_tokens=self.max_tokens, sampler=sampler, verbose=False)
            gen_s = time.monotonic() - t

            code = extract_code(raw)
            r = self.harness.render(code, quality="low", frames=4)

            trials.append(Trial(
                index=i, prompt=prompt[:180], ok=r.ok,
                error_kind=r.error_kind.value, gen_seconds=round(gen_s, 2),
                render_seconds=round(r.elapsed_s, 2), code_chars=len(code),
                n_play_calls=code.count("self.play("), code=code,
            ))
            mark = "ok " if r.ok else r.error_kind.value[:10]
            print(f"  [{i+1:>3}/{len(prompts)}] {mark:<12} gen {gen_s:5.1f}s", flush=True)

        return self._summarise(model_id, trials, out)

    @staticmethod
    def _summarise(model_id: str, trials: list[Trial], out: Path) -> dict:
        # Environment failures indict the machine, not the model.
        env = {ErrorKind.LATEX_MISSING.value, ErrorKind.MEMORY.value,
               ErrorKind.FFMPEG.value, ErrorKind.LAUNCH.value}
        scored = [t for t in trials if t.error_kind not in env]
        n_ok = sum(t.ok for t in scored)

        failures: dict[str, int] = {}
        for t in scored:
            if not t.ok:
                failures[t.error_kind] = failures.get(t.error_kind, 0) + 1

        animated = [t for t in scored if t.ok and t.n_play_calls > 0]

        summary = {
            "model": model_id,
            "n_prompts": len(trials),
            "n_scored": len(scored),
            "n_excluded_env": len(trials) - len(scored),
            "render_success_rate": round(n_ok / len(scored), 4) if scored else 0.0,
            "animated_rate": round(len(animated) / len(scored), 4) if scored else 0.0,
            "mean_play_calls": round(
                sum(t.n_play_calls for t in animated) / len(animated), 2) if animated else 0.0,
            "mean_gen_seconds": round(sum(t.gen_seconds for t in trials) / len(trials), 2),
            "failure_breakdown": dict(sorted(failures.items(), key=lambda kv: -kv[1])),
        }

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"summary": summary, "trials": [asdict(t) for t in trials]}, indent=2))
        return summary
