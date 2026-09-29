"""A local vision judge: Qwen3.5-4B (MLX, 4-bit) asks of one frame whether it
shows the beat's idea as a picture.

Gemini's free tier judges a few hundred scenes a day and the critic needs
thousands before an SFT; this runs on the Mac at ~1 s a frame. It is asked
one frame at a time -- a 4B is steadier on one image than on six -- with the
same YES/NO criteria as the Gemini critic and judge. Calibrated against
Gemini's verdicts on the scorecard runs (scripts/local_judge.py --calibrate)
before its verdicts are used for anything.

The model is 4B, not 7B: it may share the Mac with nothing else that loads
a 7B, but is small enough to run beside renders.
"""
from __future__ import annotations

import re
from pathlib import Path

MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"

ASK = ('This is a frame from an animated maths explanation of "{request}". '
       'The frame should illustrate: "{intent}". Answer YES only if the frame '
       'shows a picture (diagram, graph, geometric figure, chart or arrangement) '
       'that a viewer would learn this idea from. Answer NO if it shows only '
       'text or an equation, empty axes or a bare grid or number line, a lone '
       'shape or arrow that does not show the idea, a picture of something '
       'else, or a cluttered overlapping mess. Answer with one word: YES or NO.')


class LocalJudge:
    def __init__(self, model: str = MODEL):
        from mlx_vlm import load
        self.model, self.proc = load(model)

    def p_yes(self, request: str, intent: str, image: str | Path) -> float:
        """P(YES) against P(NO) for the first answer token, for one frame."""
        import math
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        q = ASK.format(request=request[:300], intent=intent[:300])
        prompt = apply_chat_template(self.proc, self.model.config, q, num_images=1,
                                     enable_thinking=False)
        tok = getattr(self.proc, "tokenizer", self.proc)
        if not hasattr(self, "_ids"):
            self._ids = [tok.encode(w, add_special_tokens=False)[0] for w in ("YES", "NO")]
        step = next(iter(stream_generate(self.model, self.proc, prompt, [str(image)],
                                         max_tokens=1, temperature=0.0)))
        lp = step.logprobs
        y, n = float(lp[self._ids[0]]), float(lp[self._ids[1]])
        return 1 / (1 + math.exp(n - y))

    def frame(self, request: str, intent: str, image: str | Path,
              threshold: float = 0.5) -> str:
        """'YES' or 'NO' for one frame."""
        return "YES" if self.p_yes(request, intent, image) >= threshold else "NO"
