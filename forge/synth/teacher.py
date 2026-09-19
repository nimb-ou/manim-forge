"""Teacher-model client for synthetic data generation.

A large model writes training examples; the small local model learns from them.
That is distillation, and it directly attacks this project's weakest link — the
public corpus is only ~3.7k rows and much of it is poor.

Provider-agnostic by design. Gemini, DeepSeek and OpenAI all expose an
OpenAI-compatible endpoint, so switching teacher is a base URL, not a rewrite.
Start on Gemini's free tier (1,500 requests/day, permanent) and only pay for
volume once the prompt is proven to produce data that survives the gate.

Nothing generated here is trusted. Every scene goes through the same render
gate as the scraped corpus, so a teacher that hallucinates simply produces rows
that get dropped.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """Read .env if present. Keys live there, never in the repo — .env is
    gitignored, so a key can't be committed by accident.

    Last occurrence wins, and it overrides the existing environment. Both
    matter: setdefault silently preferred a stale exported key over the correct
    one in .env, which presents as "API key not valid" for a key that is
    perfectly valid — as curl against the same file proved.
    """
    env = Path(".env")
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip("'\"")
        if v:
            os.environ[k.strip()] = v


_load_dotenv()

#: Gemini uses Google's native SDK, not the OpenAI-compatible endpoint.
#: Google migrated API keys from "Standard keys" (AIza..., rejected outright
#: since September 2026) to "Auth keys" (AQ.Ab...), and the compatibility
#: endpoint still assumes the old format — so a perfectly valid new key is
#: refused there with "Please pass a valid API key". The native SDK accepts it.
GEMINI_REST = "https://generativelanguage.googleapis.com/v1beta"

#: Rate limits are per-model, so rotating across models multiplies the usable
#: free quota. Probed against a real free-tier key: the newest models
#: (3.8-flash, the Pro tier, and the *-latest aliases pointing at them) return
#: 429 immediately, while these serve reliably. Strongest first — the rotation
#: falls back down the list rather than round-robining blindly, so most
#: generations come from the best model that will answer.
GEMINI_ROTATION = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
]

PROVIDERS = {
    "gemini": {
        "native": True,
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-flash-latest",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
    },
}

# The teacher is shown our beat format by example rather than described it —
# few-shot works far better than specification for an API it has never seen.
FEWSHOT = '''from manim import *
from forge.beats import ForgeScene, beat
from forge.primitives.grid import Board, Cell

class RookReach(ForgeScene):
    board = Board(8, 8)

    def xy(self, cell):
        x = -3.1 + (cell.col - 3.5) * 0.7
        y = (cell.row - 3.5) * 0.7 - 0.3
        return np.array([x, y, 0.0])

    @beat("Draw an 8x8 chessboard", seconds=3)
    def draw_board(self):
        self.squares = VGroup(*[
            Square(side_length=0.7, stroke_width=0, fill_opacity=1,
                   fill_color="#3D4852" if self.board.is_light(c) else "#2A313A"
                   ).move_to(self.xy(c))
            for c in self.board.cells()])
        self.play(LaggedStart(*[FadeIn(s, scale=0.6) for s in self.squares],
                              lag_ratio=0.012, run_time=2.0))

    @beat("Count the squares a rook reaches", seconds=4)
    def reach(self):
        moves = self.board.rook_moves(Cell(3, 3))        # computed, not recalled
        tiles = VGroup(*[Square(side_length=0.7, stroke_width=0, fill_opacity=0.42,
                                fill_color=YELLOW_C).move_to(self.xy(c)) for c in moves])
        label = Text(f"{len(moves)} squares", font_size=40, color=YELLOW_C).move_to([3.4, 0.6, 0])
        self.play(LaggedStart(*[FadeIn(t) for t in tiles], lag_ratio=0.05, run_time=1.6))
        self.play(Write(label), run_time=0.6)
        self.wait(0.4)
'''

SYSTEM = f"""You write 3Blue1Brown-style mathematical animations using Manim Community Edition v0.19+.

You write scenes in a BEAT format. A beat is one communicative step lasting
2-5 seconds: something appears, transforms, or gets annotated. Beats are
methods on a ForgeScene subclass, decorated with @beat("intent", seconds=N),
and they run in the order written. Mobjects assigned to `self` persist across
beats.

Here is a complete example of the format:

```python
{FEWSHOT}```

RULES, all of which matter:
1. Subclass ForgeScene, not Scene. Use @beat on every method.
2. EVERY beat must animate with self.play(...). Never write a beat that only
   calls self.add() - it renders no video.
3. Any number shown on screen must be COMPUTED in the code, never hardcoded
   from memory. Use len(), sum(), Fraction, or a loop. If you claim a
   probability is 1/16, derive it by counting.
4. Use exact values. Fraction(1,16), not 0.0625.
5. Keep objects inside the frame: x in [-7, 7], y in [-4, 4]. Put a board or
   diagram on the left and labels on the right rather than stacking them.
6. Use only real Manim CE names. Do not invent classes or keyword arguments.
7. Output ONLY one ```python code block. No explanation before or after."""


def user_prompt(topic: str, n_beats: int, length_hint: str) -> str:
    return (
        f"Write a beat-structured Manim scene explaining:\n\n  {topic}\n\n"
        f"Use about {n_beats} beats ({length_hint}).\n"
        f"Give the class a descriptive name. Make it visually clear and "
        f"genuinely explanatory, not just decorative."
    )


_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


_OPEN_FENCE = re.compile(r"^\s*```(?:python)?\s*\n", re.M)


def extract_code(text: str) -> str:
    """Pull code out of a model response, closed fence or not.

    A generation that hits the token limit is truncated mid-block, so the
    closing fence never arrives. Requiring one meant the opening ```python line
    was returned as if it were code, and every such row died with a syntax
    error on line 1.
    """
    text = text or ""
    blocks = _FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    m = _OPEN_FENCE.search(text)
    if m:
        return text[m.end():].strip().removesuffix("```").strip()
    return text.strip()


@dataclass
class Teacher:
    provider: str = "gemini"
    model: str | None = None
    temperature: float = 0.7      # some diversity: identical scenes teach nothing
    last_model_used: str = ""

    def __post_init__(self):
        cfg = PROVIDERS[self.provider]
        key = os.environ.get(cfg["key_env"])
        if not key:
            raise RuntimeError(
                f"Set {cfg['key_env']} in your environment or .env file.")
        self.model = self.model or cfg["model"]
        self.native = cfg.get("native", False)
        if self.native:
            # Raw REST rather than the google-genai SDK. Google's new "Auth
            # keys" (AQ.Ab...) are rejected by both the OpenAI-compatibility
            # endpoint and the current SDK, while the REST endpoint accepts
            # them without complaint. One less dependency, and it demonstrably
            # works.
            self._key = key
            self._client = None
        else:
            from openai import OpenAI
            self._client = OpenAI(api_key=key, base_url=cfg["base_url"])

    def _gemini_rest(self, user: str, max_tokens: int, attempts: int = 5) -> str:
        """Try each model in the rotation before giving up.

        A 429 means *this model* is exhausted, not that the key is. Moving to
        the next model recovers immediately where sleeping would waste a minute
        and then fail anyway.
        """
        import json as _json
        import time as _time
        import urllib.error
        import urllib.request
        body = _json.dumps({
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "generationConfig": {"temperature": self.temperature,
                                 "maxOutputTokens": max_tokens},
        }).encode()
        # 503 (overloaded) and 429 (rate limited) are routine on a free tier
        # and are not failures — they mean "try again shortly", or better,
        # "ask a different model".
        models = [self.model] + [m for m in GEMINI_ROTATION if m != self.model] \
            if self.provider == "gemini" else [self.model]

        last = None
        data = None
        for attempt in range(attempts):
            for m in models:
                req = urllib.request.Request(
                    f"{GEMINI_REST}/models/{m}:generateContent",
                    data=body, headers={"x-goog-api-key": self._key,
                                        "Content-Type": "application/json"})
                try:
                    data = _json.load(urllib.request.urlopen(req, timeout=240))
                    self.last_model_used = m
                    break
                except urllib.error.HTTPError as e:
                    last = e
                    if e.code not in (429, 500, 502, 503, 504):
                        raise
            if data is not None:
                break
            _time.sleep(min(2 ** attempt * 3, 60))
        if data is None:
            raise last
        cands = data.get("candidates") or []
        if not cands:
            return ""
        parts = cands[0].get("content", {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts)

    @classmethod
    def list_models(cls, provider: str = "gemini") -> list[str]:
        """What this API key can actually reach.

        Chat product names and API model ids drift apart, so ask the endpoint
        rather than guessing from what the web UI offers.
        """
        cfg = PROVIDERS[provider]
        key = os.environ.get(cfg["key_env"])
        if not key:
            raise RuntimeError(f"Set {cfg['key_env']} first.")
        if cfg.get("native"):
            import json as _json
            import urllib.request
            req = urllib.request.Request(
                f"{GEMINI_REST}/models?pageSize=200",
                headers={"x-goog-api-key": key})
            data = _json.load(urllib.request.urlopen(req, timeout=60))
            return sorted(
                m["name"].removeprefix("models/") for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            )
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=cfg["base_url"])
        return sorted(m.id for m in client.models.list())

    def generate(self, topic: str, n_beats: int, length_hint: str,
                 max_tokens: int = 6000) -> str:
        user = user_prompt(topic, n_beats, length_hint)
        if self.native:
            return extract_code(self._gemini_rest(user, max_tokens))
        r = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return extract_code(r.choices[0].message.content)
