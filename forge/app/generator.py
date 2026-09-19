"""Generator interface — one contract, several backends.

The website will start on the Gemini free tier while the local model trains,
then move to the fine-tuned model, and probably end up using both: local first,
escalating to the API when the local model fails twice.

That only stays cheap if nothing downstream knows which is which. So the app,
the repair loop and the evaluation all talk to this interface, and swapping
backends is a constructor argument rather than a rewrite.

The escalating backend is the interesting one. Every escalation is a case the
local model could not handle and a stronger model could — which is precisely
the highest-value training example available. Using the API makes the local
model better rather than competing with it.
"""

from __future__ import annotations

from typing import Protocol

#: A complete beat-structured scene is 2-4k tokens. Anything tighter truncates
#: mid-function and presents as a syntax error, which reads as a bad model
#: rather than a bad budget.
MAX_TOKENS = 4500


class Generator(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
        ...


class LocalMLX:
    """The fine-tuned model running on Apple Silicon."""

    def __init__(self, model_id: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                 adapter_path: str | None = None, temperature: float = 0.0):
        from mlx_lm import load
        self.name = f"local:{model_id.split('/')[-1]}"
        kw = {"adapter_path": adapter_path} if adapter_path else {}
        self.model, self.tok = load(model_id, **kw)
        self.temperature = temperature

    def complete(self, system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        chat = self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False)
        return generate(self.model, self.tok, prompt=chat, max_tokens=max_tokens,
                        sampler=make_sampler(temp=self.temperature), verbose=False)


class RemoteGemini:
    """The free-tier teacher, reused as a server backend."""

    def __init__(self, model: str = "gemini-3.7-flash", temperature: float = 0.4):
        from forge.synth.teacher import Teacher
        self._t = Teacher(provider="gemini", model=model, temperature=temperature)
        self.name = f"gemini:{model}"

    def complete(self, system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
        import forge.synth.teacher as T
        original = T.SYSTEM
        try:
            T.SYSTEM = system          # the client reads module-level SYSTEM
            return self._t._gemini_rest(user, max_tokens)
        finally:
            T.SYSTEM = original


class Escalating:
    """Try the cheap generator; fall back to the strong one.

    ``escalations`` accumulates every prompt the primary could not handle.
    Those are the rows worth training on next.
    """

    def __init__(self, primary: Generator, fallback: Generator):
        self.primary, self.fallback = primary, fallback
        self.name = f"{primary.name}->{fallback.name}"
        self.escalations: list[str] = []

    def complete(self, system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
        try:
            return self.primary.complete(system, user, max_tokens)
        except Exception:
            self.escalations.append(user)
            return self.fallback.complete(system, user, max_tokens)

    def escalate(self, system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
        """Explicit escalation after the primary's output failed to render."""
        self.escalations.append(user)
        return self.fallback.complete(system, user, max_tokens)
