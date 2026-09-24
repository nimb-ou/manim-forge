"""The two-stage pipeline as a library: request in, events out, video at the end.

    from forge.app.pipeline import Options, SwapHost, run
    host = SwapHost("adapters/mlx-planner3", "adapters/mlx-coder2")
    result = run("why the harmonic series diverges", host, print, Options())

Three callers used to carry their own copy of this -- the eval runner, the
demo, and now the server -- and two copies of the coder prompt had already
drifted once. The eval runner keeps its measurement-only extras (targeted
repair, metrics) but takes its prompts, decoding and planning from here.

Every stage reports through ``emit(event)``, a plain dict with a ``stage``
key, so the terminal demo prints them and the server streams them.
"""
from __future__ import annotations

import ast
import gc
import re
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from forge.app.twostage import (Beat, assemble, beat_prompt, extract_code,
                                failing_beat, intent_key, missing_names,
                                parse_plan, parsing_prefix, prelude_prompt,
                                prune_all)

MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

PLAN_SYSTEM = (
    "You plan 3Blue1Brown-style mathematical animations, a few beats at a "
    "time. You are given the request and the beats already planned. Continue "
    "the arc: write the next beats and nothing else, in the form\n"
    "  N. [seconds] intent -- narration\n"
    "keeping the numbering running. Build a real explanatory arc rather than "
    "a list of topics. When the explanation is complete, write END on its "
    "own line after the last beat."
)
CODE_SYSTEM = (
    "You write one beat of a 3Blue1Brown-style Manim scene. You are given the "
    "whole request, the beats already on screen, and the helpers the scene "
    "defines. Write only the code for the beat you are asked for, as "
    "statements at method-body level -- no class, no def, no imports. "
    "NAMES IN SCOPE lists what earlier beats already built: reuse those "
    "rather than rebuilding them. Anything else you use you must "
    "CONSTRUCT in this beat before you animate it -- `self.play(Create(dot))` "
    "is wrong unless a line above it makes `dot`."
)

def _kit_api() -> str:
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parents[1] / "kit" / "kit.py").read_text()
    return src.split('KIT_API = """\\\n', 1)[1].split('"""', 1)[0]


#: Kit mode: the coder writes beats as calls to forge/kit, which draw the
#: picture themselves. The model has not been trained on the kit, so the API
#: and two worked beats are in the prompt.
CODE_SYSTEM_KIT = (
    "You write one beat of a 3Blue1Brown-style animation using the Forge kit, "
    "a library of building blocks that draw and animate the picture for you. "
    "Write only the statements for the beat you are asked for -- no class, no "
    "def, no imports. Show the idea with the kit's pictures; use text only "
    "through stage.title, stage.caption, stage.label and stage.equation. "
    "NAMES IN SCOPE lists what earlier beats built; reuse them. Anything else "
    "you use, make in this beat first. You may also use any Manim call "
    "(self.play(x.animate...), Transform, ...) on the objects the kit returns."
    "\n\nTHE KIT\n" + _kit_api() +
    "\nEXAMPLE -- intent: Two vectors add tip to tail\n"
    "p = plane(stage)\n"
    "v = vector(stage, p, (2, 1), YELLOW, \"v\")\n"
    "w = vector(stage, p, (1, 2), BLUE, \"w\")\n"
    "stage.caption(\"Slide w so its tail sits on v's tip\")\n"
    "self.play(w.animate.shift(p.c2p(2, 1) - p.c2p(0, 0)))\n"
    "s = vector(stage, p, (3, 3), GREEN, \"v + w\")\n"
    "\nEXAMPLE -- intent: The slope of x squared at every point\n"
    "ax = axes(stage, x_range=(-1, 3), y_range=(-1, 9))\n"
    "f = lambda x: x ** 2\n"
    "g = graph(stage, ax, f, label=\"x^2\")\n"
    "tangent(stage, ax, f, 0.2, 2.5)\n"
    "stage.caption(\"The slope grows with x: it is 2x\")\n"
)

#: Planner decoding. Sampled: on eight hard titles planner v2 wrote 3.8 beats
#: before its first repeated intent under greedy decoding and 22.6 at
#: temperature 0.5 with a 1.1 repetition penalty. The coder stays greedy.
PLAN_DECODE = {"temp": 0.5, "rep_penalty": 1.1}

Emit = Callable[[dict], None]


def load(adapter: str | None):
    from mlx_lm import load as mlx_load
    return mlx_load(MODEL, **({"adapter_path": adapter} if adapter else {}))


def ask(model, tok, system: str, user: str, max_tokens: int,
        temp: float = 0.0, rep_penalty: float = 0.0) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    chat = tok.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        add_generation_prompt=True, tokenize=False)
    procs = (make_logits_processors(repetition_penalty=rep_penalty,
                                    repetition_context_size=256)
             if rep_penalty else None)
    return generate(model, tok, prompt=chat, max_tokens=max_tokens,
                    sampler=make_sampler(temp=temp,
                                         top_p=0.95 if temp else 0.0),
                    logits_processors=procs, verbose=False)


def plan(model, tok, request: str, stride: int, max_beats: int,
         on_beat: Callable[[Beat], None] | None = None,
         decode: dict | None = None) -> list[Beat]:
    """Feed the planner its own output until END, a repeat, or the cap.

    A repeated intent -- compared without its counting, see intent_key --
    ends the arc: planner v2 looped outright, planner v3 looped by counting.
    """
    decode = PLAN_DECODE if decode is None else decode
    beats: list[Beat] = []
    for _ in range(max_beats // stride + 1):
        shown = "\n".join(
            f"{b.n}. [{f'{b.seconds:g}s' if b.seconds else '?'}] {b.intent}"
            for b in beats[-12:]) or "  (nothing yet — open the explanation)"
        reply = ask(model, tok, PLAN_SYSTEM,
                    f"REQUEST\n{request}\n\nBEATS SO FAR\n{shown}\n\n"
                    f"Write the next {stride} beat(s), numbered from "
                    f"{len(beats) + 1}.", max_tokens=700, **decode)
        new, ended = parse_plan(reply, limit=stride)
        new = [b for b in new if b.n > len(beats)]
        seen = {intent_key(b.intent) for b in beats}
        for k, b in enumerate(new):
            key = intent_key(b.intent)
            if key in seen:
                new, ended = new[:k], True
                break
            seen.add(key)
        if not new:
            break
        for b in new[: max_beats - len(beats)]:
            beats.append(b)
            if on_beat:
                on_beat(b)
        if ended or len(beats) >= max_beats:
            break
    return beats


def write_beat(model, tok, request: str, beats: list[Beat], j: int,
               bodies: list[str], max_tokens: int,
               system: str = CODE_SYSTEM) -> tuple[str, str]:
    """One beat's code: two tries to parse, then its parsing prefix.

    Returns (body, note); an empty body means the beat was dropped.
    """
    cand = ""
    for _ in range(2):
        cand = extract_code(ask(model, tok, system,
                                beat_prompt(request, beats, j, bodies),
                                max_tokens=max_tokens))
        try:
            ast.parse(textwrap.dedent(cand))
            return cand, ""
        except SyntaxError:
            continue
    head = parsing_prefix(cand)
    # "Animates something" means it calls something: a raw beat's self.play,
    # or a kit block, which plays its own animation (and so never contains
    # the string "self.play(").
    if head.strip() and any(isinstance(n, ast.Call)
                            for n in ast.walk(ast.parse(head))):
        return head, "truncated; kept the lines before the cut"
    return "", "did not parse twice; dropped"


# -- models --------------------------------------------------------------------

class Sequential:
    """One adapter in memory at a time: load the planner, drop it, load the
    coder. Slow (a load per stage) but needs 4.5 GB, not 9."""

    def __init__(self, planner: str | None, coder: str | None):
        self.paths = {"planner": planner, "coder": coder}
        self.current: str | None = None
        self.model = self.tok = None

    def use(self, which: str):
        if self.current != which:
            self.model = self.tok = None
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:                                 # noqa: BLE001
                pass
            self.model, self.tok = load(self.paths[which])
            self.current = which
        return self.model, self.tok


class SwapHost:
    """One base model, two LoRA adapters, swapped in place.

    Both adapters are rank 16 over the same seven projections at scale 2.0,
    so the LoRA layers mlx-lm builds for one fit the other's weights, and a
    swap is ``load_weights`` of a 160 MB file instead of reloading 4 GB.
    ``strict=False`` is a silent no-op when keys do not match -- the
    conversion bug this project already hit once -- so every swap checks
    that a tensor actually changed to the file's value.
    """

    def __init__(self, planner: str, coder: str):
        import json
        cfgs = []
        for p in (planner, coder):
            c = json.loads((Path(p) / "adapter_config.json").read_text())
            lp = c["lora_parameters"]
            cfgs.append((c["num_layers"], lp["rank"], lp["scale"],
                         tuple(sorted(lp["keys"]))))
        if cfgs[0] != cfgs[1]:
            raise ValueError(f"adapters differ in shape, cannot swap: {cfgs}")
        self.paths = {"planner": planner, "coder": coder}
        self.model, self.tok = load(planner)
        self.current = "planner"
        import mlx.core as mx
        self._weights = {k: mx.load(str(Path(p) / "adapters.safetensors"))
                         for k, p in self.paths.items()}
        self._probe = sorted(self._weights["coder"])[0]

    def _param(self, name: str):
        obj = self.model
        for part in name.split("."):
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        return obj

    def use(self, which: str):
        if self.current != which:
            import mlx.core as mx
            w = self._weights[which]
            self.model.load_weights(list(w.items()), strict=False)
            mx.eval(self.model.parameters())
            got = self._param(self._probe)
            if not bool(mx.allclose(got, w[self._probe])):
                raise RuntimeError(f"adapter swap to {which} did not take "
                                   f"({self._probe} unchanged)")
            self.current = which
        return self.model, self.tok


# -- the run -------------------------------------------------------------------

@dataclass
class Options:
    max_beats: int = 12
    stride: int = 6
    beat_tokens: int = 1400
    quality: str = "medium"
    setup: bool = True
    salvage: bool = True
    kit: bool = False


@dataclass
class Result:
    request: str
    beats: list[Beat]
    bodies: list[str]
    code: str = ""
    ok: bool = False
    video: str | None = None
    duration: float | None = None
    error: str = ""
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0


def run(request: str, host, emit: Emit, opts: Options | None = None,
        harness=None) -> Result:
    """Plan, implement, assemble (with set-up and salvage), render."""
    opts = opts or Options()
    t0 = time.time()
    notes: list[str] = []

    def note(msg: str, **kw) -> None:
        notes.append(msg)
        emit({"stage": "assemble", "note": msg, **kw})

    # 1. plan
    emit({"stage": "plan", "status": "start"})
    pm, ptok = host.use("planner")
    beats = plan(pm, ptok, request, opts.stride, opts.max_beats,
                 on_beat=lambda b: emit({
                     "stage": "plan", "beat": {"n": b.n, "seconds": b.seconds,
                                               "intent": b.intent,
                                               "narration": b.narration}}))
    emit({"stage": "plan", "status": "done", "n": len(beats),
          "elapsed": round(time.time() - t0, 1)})
    if not beats:
        emit({"stage": "done", "ok": False, "error": "the planner wrote no beats"})
        return Result(request, [], [], error="no beats")

    # 2. implement
    emit({"stage": "code", "status": "start"})
    cm, ctok = host.use("coder")
    system = CODE_SYSTEM_KIT if opts.kit else CODE_SYSTEM
    kit = opts.kit
    bodies: list[str] = []
    for j, b in enumerate(beats):
        body, why = write_beat(cm, ctok, request, beats, j, bodies,
                               opts.beat_tokens, system)
        bodies.append(body)
        emit({"stage": "code", "beat": b.n, "intent": b.intent, "code": body,
              "note": why})
    emit({"stage": "code", "status": "done",
          "elapsed": round(time.time() - t0, 1)})

    # 3. assemble: set-up, then statement-level, then beat-level salvage
    emit({"stage": "assemble", "status": "start"})
    missing = missing_names(beats, bodies, kit=kit)
    if opts.setup and missing and any(x.strip() for x in bodies):
        pre = parsing_prefix(extract_code(ask(
            cm, ctok, system, prelude_prompt(request, bodies, missing),
            max_tokens=opts.beat_tokens)))
        if pre.strip():
            first = next(k for k, x in enumerate(bodies) if x.strip())
            trial = list(bodies)
            trial[first] = textwrap.dedent(pre).strip() + "\n" + \
                textwrap.dedent(trial[first])
            if len(missing_names(beats, trial, kit=kit)) < len(missing):
                bodies = trial
                note(f"set-up built {', '.join(missing[:6])}", code=pre)
    live = sum(1 for x in bodies if x.strip())
    if opts.salvage:
        trial, n, missing = prune_all(beats, bodies, kit=kit)
        drawing = sum(1 for x in trial if x.strip())
        if n and assemble(beats, trial, kit=kit).ok and 2 * drawing >= live:
            bodies = trial
            note(f"removed {n} lines using {', '.join(missing[:4])}, "
                 f"which nothing builds")
        for _ in range(len(bodies)):
            missing = missing_names(beats, bodies, kit=kit)
            if not missing:
                break
            hit = [j for j, x in enumerate(bodies) if x.strip() and any(
                re.search(rf"\b{re.escape(m.removeprefix('self.'))}\b", x)
                for m in missing)]
            if not hit:
                break
            for j in hit:
                bodies[j] = ""
            note(f"dropped beats {', '.join(str(j + 1) for j in hit)} "
                 f"(they use {', '.join(missing[:4])})")
    asm = assemble(beats, bodies, kit=kit)
    kept = sum(1 for x in bodies if x.strip())
    if not asm.ok or kept * 2 < live:
        why = asm.problems[0] if asm.problems else \
            f"only {kept} of {live} beats survived"
        emit({"stage": "done", "ok": False, "error": why, "code": asm.code})
        return Result(request, beats, bodies, asm.code, error=why,
                      notes=notes, seconds=time.time() - t0)
    emit({"stage": "assemble", "status": "done", "kept": kept,
          "of": len(beats), "code": asm.code})

    # 4. render, dropping a beat the traceback blames, three times at most
    if harness is None:
        from forge.harness import RenderHarness
        root = Path(__file__).resolve().parents[2]
        harness = RenderHarness(python_bin=str(root / ".venv" / "bin" / "python"),
                                cache_dir=str(root / "data" / "frames"),
                                timeout=600, store_video=True)
    emit({"stage": "render", "status": "start", "quality": opts.quality})
    res = harness.render(asm.code, quality=opts.quality, use_cache=False)
    for _ in range(3 if opts.salvage else 0):
        if res.ok:
            break
        k = failing_beat(asm.code, res.stderr or "")
        if k is None or not bodies[k - 1].strip():
            break
        bodies[k - 1] = ""
        # The dropped beat may have built names later beats use.
        bodies, _, _ = prune_all(beats, bodies, kit=kit)
        trial = assemble(beats, bodies, kit=kit)
        if not trial.ok or 2 * sum(1 for x in bodies if x.strip()) < live:
            break
        asm = trial
        emit({"stage": "render", "note": f"beat {k} failed at runtime "
              f"({res.error_kind.value}); re-rendering without it"})
        notes.append(f"runtime: dropped beat {k}")
        res = harness.render(asm.code, quality=opts.quality, use_cache=False)
    out = Result(request, beats, bodies, asm.code, ok=res.ok,
                 video=res.video_path if res.ok else None,
                 duration=res.duration_s, notes=notes,
                 error="" if res.ok else res.error_kind.value,
                 seconds=time.time() - t0)
    emit({"stage": "done", "ok": res.ok, "video": out.video,
          "duration": res.duration_s, "error": out.error,
          "stderr": "" if res.ok else (res.stderr or "")[-1500:],
          "code": asm.code, "elapsed": round(out.seconds, 1)})
    return out
