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
                                intent_key, missing_names,
                                parse_plan, parsing_prefix, prelude_prompt,
                                prune_all, runtime_salvage)

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
    "p = draw_plane(stage)\n"
    "v = draw_vector(stage, p, (2, 1), YELLOW, \"v\")\n"
    "w = draw_vector(stage, p, (1, 2), BLUE, \"w\")\n"
    "stage.caption(\"Slide w so its tail sits on v's tip\")\n"
    "self.play(w.animate.shift(p.c2p(2, 1) - p.c2p(0, 0)))\n"
    "s = draw_vector(stage, p, (3, 3), GREEN, \"v + w\")\n"
    "\nEXAMPLE -- intent: The slope of x squared at every point\n"
    "ax = draw_axes(stage, x_range=(-1, 3), y_range=(-1, 9))\n"
    "f = lambda x: x ** 2\n"
    "g = plot_graph(stage, ax, f, label=\"x^2\")\n"
    "slide_tangent(stage, ax, f, 0.2, 2.5)\n"
    "stage.caption(\"The slope grows with x: it is 2x\")\n"
)

#: The kit coder's own prompt, once trained. It learns the kit from its data,
#: so the 1,293-token API list stays with the teacher: rows drop from ~1,850
#: tokens to ~600, three times faster to train and to run. Training rows are
#: rewritten with this prompt (scripts/build_kit_dataset.py).
CODE_SYSTEM_KIT_TRAINED = (
    "You write one beat of a 3Blue1Brown-style animation with the Forge kit: "
    "its blocks (draw_plane, draw_vector, apply_matrix, draw_axes, plot_graph, "
    "slide_tangent, riemann_refine, ...) take the stage first, draw and "
    "animate the picture, and return what they made. `stage = Stage(self)` "
    "exists. Write only this beat's statements -- no class, no def, no "
    "imports. Show the idea with pictures; text only through stage.title, "
    "stage.caption, stage.label and stage.equation. Reuse NAMES IN SCOPE; "
    "make anything else in this beat first."
)

#: Planner decoding. Sampled: on eight hard titles planner v2 wrote 3.8 beats
#: before its first repeated intent under greedy decoding and 22.6 at
#: temperature 0.5 with a 1.1 repetition penalty. The coder stays greedy.
PLAN_DECODE = {"temp": 0.5, "rep_penalty": 1.1}

Emit = Callable[[dict], None]


def load(adapter: str | None, base: str = MODEL):
    from mlx_lm import load as mlx_load
    return mlx_load(base, **({"adapter_path": adapter} if adapter else {}))


_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S)


def ask(model, tok, system: str, user: str, max_tokens: int,
        temp: float = 0.0, rep_penalty: float = 0.0, think: bool = False) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    # Qwen3.x reasons in a <think> block unless told not to; the scene is the
    # answer. Other templates ignore the flag.
    chat = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                   tokenize=False, enable_thinking=think)
    procs = (make_logits_processors(repetition_penalty=rep_penalty,
                                    repetition_context_size=256)
             if rep_penalty else None)
    out = generate(model, tok, prompt=chat, max_tokens=max_tokens,
                    sampler=make_sampler(temp=temp,
                                         top_p=0.95 if temp else 0.0),
                    logits_processors=procs, verbose=False)
    # With thinking on, the template may open <think> in the prompt itself,
    # so the reply holds only the closing tag.
    if "</think>" in out:
        out = out.split("</think>")[-1]
    return _THINK.sub("", out).strip()


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
               system: str = CODE_SYSTEM, relevance: bool = False,
               exemplar: bool = False, signatures: bool = False) -> tuple[str, str]:
    """One beat's code: two tries to parse, then its parsing prefix.

    With ``relevance`` (kit only), the prompt names the blocks of the
    subject the request and beat are about, and a beat drawn only with
    another subject's blocks -- a vector for a neural network -- is sampled
    again, twice at most, keeping the first that fits.

    Returns (body, note); an empty body means the beat was dropped.
    """
    from forge.kit.families import hint, relevant
    user = beat_prompt(request, beats, j, bodies)
    text = f"{beats[j].intent} {beats[j].narration or ''} {request}"
    if relevance and (h := hint(text, signatures=signatures)):
        user += "\n" + h
    if exemplar:
        from forge.kit.exemplars import example
        if ex := example(f"{request} {beats[j].intent}"):
            user += "\n\n" + ex
    cand = ""
    parsed: str | None = None
    for k in range(4 if relevance else 2):
        cand = extract_code(ask(model, tok, system, user, max_tokens=max_tokens,
                                temp=0.7 if k else 0.0))
        try:
            ast.parse(textwrap.dedent(cand))
        except SyntaxError:
            continue
        if not relevance or relevant(cand, text) is not False:
            return cand, "" if k == 0 else f"resampled {k}x for a fitting picture"
        parsed = parsed or cand
    if parsed is not None:
        return parsed, "no fitting picture in 4 samples; kept the first"
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
    kit_trained: bool = True        # the short prompt; False = prompt-only
    relevance: bool = False         # subject hint + resample off-subject beats
    exemplar: bool = False          # a hand-written scene for a similar request
    mark_beats: bool = False        # record each beat's end time (kit only)
    narrate: bool = False           # speak each beat's narration (forge/app/voice.py)
    # The relevance hint with each block's call: 45%/37% (short/held-out) vs
    # 58%/39% for names only, on identical plans -- off.
    signatures: bool = False


def _failure(stderr: str) -> str:
    """The last exception message and the scene line that raised it, from a
    manim traceback (rich-formatted)."""
    lines = [re.sub(r"[│╭╮╰╯─]+", " ", x).strip() for x in stderr.splitlines()]
    err = next((x for x in reversed(lines)
                if re.match(r"^[A-Za-z_.]*(Error|Exception)\b", x)), "")
    src = next((x for x in reversed(lines) if x.startswith("❱")), "")
    src = re.sub(r"^❱\s*\d+\s*", "", src).strip()
    return (f"{err} at `{src}`" if src else err)[:300]


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
    beat_ends: list[float] = field(default_factory=list)
    layout: list[int] = field(default_factory=list)   # kit.layout_issues per beat
    issues: list[tuple[int, str]] = field(default_factory=list)  # (beat, what the kit saw)
    reply: str = ""          # the model's own text, for the one shot (training rows)
    failure: str = ""        # a failed render's error and line, for the critic


def run(request: str, host, emit: Emit, opts: Options | None = None,
        harness=None, beats: list[Beat] | None = None) -> Result:
    """Plan, implement, assemble (with set-up and salvage), render."""
    opts = opts or Options()
    t0 = time.time()
    notes: list[str] = []

    def note(msg: str, **kw) -> None:
        notes.append(msg)
        emit({"stage": "assemble", "note": msg, **kw})

    # 1. plan
    emit({"stage": "plan", "status": "start"})
    if beats is None:            # a given plan (evaluations) skips the planner
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
    system = (CODE_SYSTEM_KIT_TRAINED if opts.kit_trained else CODE_SYSTEM_KIT) \
        if opts.kit else CODE_SYSTEM
    kit = opts.kit
    bodies: list[str] = []
    for j, b in enumerate(beats):
        body, why = write_beat(cm, ctok, request, beats, j, bodies,
                               opts.beat_tokens, system,
                               relevance=opts.relevance and kit,
                               exemplar=opts.exemplar and kit,
                               signatures=opts.signatures)
        bodies.append(body)
        emit({"stage": "code", "beat": b.n, "intent": b.intent, "code": body,
              "note": why})
    emit({"stage": "code", "status": "done",
          "elapsed": round(time.time() - t0, 1)})
    return finish(request, beats, bodies, cm, ctok, system, opts, emit,
                  harness, t0, notes)


def finish(request: str, beats: list[Beat], bodies: list[str], cm, ctok,
           system: str, opts: Options, emit: Emit, harness=None,
           t0: float | None = None, notes: list[str] | None = None) -> Result:
    """Assemble (set-up, statement- then beat-level salvage) and render.

    Shared by the two-stage run and the one-shot generator
    (forge/app/oneshot.py), which arrives here with every beat written.
    """
    t0 = time.time() if t0 is None else t0
    notes = [] if notes is None else notes
    kit = opts.kit

    def note(msg: str, **kw) -> None:
        notes.append(msg)
        emit({"stage": "assemble", "note": msg, **kw})

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

    # 4. render; on a runtime error drop the statement the traceback blames
    # (a second failure in that beat drops the beat), five times at most
    if harness is None:
        from forge.harness import RenderHarness
        root = Path(__file__).resolve().parents[2]
        harness = RenderHarness(python_bin=str(root / ".venv" / "bin" / "python"),
                                cache_dir=str(root / "data" / "frames"),
                                timeout=600, store_video=True)
    mark = (opts.mark_beats or opts.narrate) and kit
    voiced: list = []
    if opts.narrate and kit:
        try:
            from forge.app.voice import speak
            import tempfile
            voiced = speak([b.narration or "" for b in beats],
                           Path(tempfile.mkdtemp(prefix="forge-voice-")))
        except Exception as exc:                              # noqa: BLE001
            notes.append(f"narration skipped ({type(exc).__name__}: {exc})")
            voiced = []
    hold = [(d + 0.35 if d else 0.0) for _, d in voiced] or [0.0] * len(beats)

    def marked(bs: list[str]) -> list[str]:
        if not mark:
            return bs
        return [b + (f"\nstage.mark(min_seconds={hold[j]:.2f})" if hold[j]
                     else "\nstage.mark()") if b.strip() else b
                for j, b in enumerate(bs)]
    if mark:
        asm = assemble(beats, marked(bodies), kit=kit)
    emit({"stage": "render", "status": "start", "quality": opts.quality})
    res = harness.render(asm.code, quality=opts.quality, use_cache=False)
    tried: set[int] = set()
    for _ in range(5 if opts.salvage else 0):
        if res.ok:
            break
        fix = runtime_salvage(beats, bodies, asm.code, res.stderr or "",
                              kit=kit, tried=tried)
        if fix is None:
            break
        trial_bodies, what = fix
        trial = assemble(beats, marked(trial_bodies), kit=kit)
        if not trial.ok or 2 * sum(1 for x in trial_bodies if x.strip()) < live:
            break
        bodies, asm = trial_bodies, trial
        emit({"stage": "render", "note": f"{what}; re-rendering"})
        notes.append(what)
        res = harness.render(asm.code, quality=opts.quality, use_cache=False)
    got = re.findall(r"BEAT_ENDS \[([^\]]*)\]", res.stderr or "")
    ends = [float(x) for x in got[-1].split(",") if x.strip()] if got else []
    lay = re.findall(r"LAYOUT \[([^\]]*)\]", res.stderr or "")
    layout = [int(x) for x in lay[-1].split(",") if x.strip()] if lay else []
    # The kit numbers beats by the marks it has passed, which skip dropped
    # beats; map them back to the scene's own beat numbers.
    live_n = [j + 1 for j, b in enumerate(bodies) if b.strip()]
    issues = []
    for k, what in re.findall(r"^ISSUE (\d+) (.*)$", res.stderr or "", re.M):
        k = int(k)
        issues.append((live_n[k - 1] if 0 < k <= len(live_n) else k, what.strip()))
    video = res.video_path if res.ok else None
    if video and voiced and ends:
        from forge.app.voice import mux
        live = [j for j, b in enumerate(bodies) if b.strip()]
        starts = [0.0] + ends[:-1]
        clips = [(voiced[j][0], t) for j, t in zip(live, starts)]
        dest = Path(video).with_name(Path(video).stem + "_voiced.mp4")
        if mux(video, clips, dest):
            video = str(dest)
        else:
            notes.append("narration could not be mixed in; silent video")
    out = Result(request, beats, bodies, asm.code, ok=res.ok, beat_ends=ends,
                 layout=layout, issues=issues,
                 failure="" if res.ok else _failure(res.stderr or ""),
                 video=video,
                 duration=res.duration_s, notes=notes,
                 error="" if res.ok else res.error_kind.value,
                 seconds=time.time() - t0)
    emit({"stage": "done", "ok": res.ok, "video": out.video,
          "duration": res.duration_s, "error": out.error,
          "stderr": "" if res.ok else (res.stderr or "")[-1500:],
          "code": asm.code, "elapsed": round(out.seconds, 1)})
    return out
