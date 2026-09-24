"""Training data for a kit-trained coder: teacher-written kit beats that render.

The coder has never seen the Forge kit; kit mode is prompt-only. If the kit
makes pictures, the coder should be trained to write it. For each plan (real
3Blue1Brown narration arcs first, then referee-checked synthetic arcs), a
teacher writes every beat as kit calls in one pass; the scene is assembled
with the kit and rendered; a beat that fails at runtime is dropped (the
traceback names it) and the rest re-rendered, up to three times. Only
scenes that render, and only beats that draw something, become rows:

    system = CODE_SYSTEM_KIT
    user   = beat_prompt(request, beats, j, bodies)   -- the inference prompt
    answer = the beat's code

Resumable, one arc per teacher call.

    ./.venv/bin/python -u scripts/synth_kit_beats.py --limit 300
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from forge.app.pipeline import CODE_SYSTEM_KIT  # noqa: E402
from forge.app.twostage import (Beat, assemble, beat_prompt,  # noqa: E402
                                failing_beat, missing_names, parse_plan,
                                parsing_prefix, prune_statements)
from scorecard import visual  # noqa: E402

D = ROOT / "data" / "planner"
OUT = ROOT / "data" / "kit" / "kit_beats.jsonl"
SCENES = ROOT / "data" / "kit" / "kit_scenes.jsonl"

TEACHER = """You are writing a 3Blue1Brown-style animation with the Forge kit.
{api}
Write the code for EVERY beat of the plan below, in order, as statements
that run inside construct() -- no class, no def, no imports. `stage =
Stage(self)` already exists. Each beat must SHOW its idea with the kit's
pictures (planes, vectors, graphs, areas, shapes, networks...); use text
only via stage.title / stage.caption / stage.label / stage.equation, and
at most one caption per beat. Reuse objects earlier beats made; clear the
stage (stage.clear()) when the picture changes subject. Keep each beat to
3-12 lines. Plain Manim on the kit's objects (self.play(x.animate...)) is
fine.

Answer with one block per beat, exactly:
### beat 1
<code>
### beat 2
<code>
...

REQUEST
{request}

PLAN
{plan}
"""
SPLIT = re.compile(r"^###\s*beat\s*(\d+)\s*$", re.M)


def arcs(limit: int) -> list[tuple[str, str, list[Beat]]]:
    """(id, request, beats): real narration arcs, then checked synthetic."""
    out = []
    for path in (D / "plan.jsonl", D / "plan_synth.jsonl"):
        if not path.exists():
            continue
        ok = None
        if path.name == "plan_synth.jsonl":
            j = D / "arc_judgements.jsonl"
            ok = {json.loads(l)["id"] for l in j.read_text().splitlines()
                  if l.strip() and json.loads(l)["verdict"] == "OK"} \
                if j.exists() else set()
        for l in path.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            rid = r["meta"].get("id") or hashlib.sha256(
                r["messages"][1]["content"].encode()).hexdigest()[:12]
            if ok is not None and rid not in ok:
                continue
            beats, _ = parse_plan(r["messages"][2]["content"], limit=60)
            if len(beats) >= 4:
                out.append((rid, r["messages"][1]["content"], beats[:10]))
    return out[:limit]


def bodies_from(reply: str, n: int) -> list[str]:
    parts = SPLIT.split(reply)
    got: dict[int, str] = {}
    for k in range(1, len(parts) - 1, 2):
        idx = int(parts[k])
        code = re.sub(r"^```\w*\n|```\s*$", "", parts[k + 1].strip(), flags=re.M)
        got[idx] = parsing_prefix(code)
    return [got.get(i + 1, "") for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="mistral")
    ap.add_argument("--model", default="mistral-medium-latest")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--pause", type=float, default=2.0)
    a = ap.parse_args()

    import manim
    mob = {n for n in dir(manim) if isinstance(getattr(manim, n), type)
           and issubclass(getattr(manim, n), manim.Mobject)}
    from forge.app.pipeline import _kit_api
    from forge.harness import RenderHarness
    from forge.synth.teacher import Teacher
    teacher = Teacher(provider=a.provider, model=a.model)
    h = RenderHarness(python_bin=str(ROOT / ".venv" / "bin" / "python"),
                      cache_dir=str(ROOT / "data" / "frames"), timeout=240)

    def _impatient(signum, frame):                            # noqa: ANN001
        raise TimeoutError("teacher did not answer in time")
    signal.signal(signal.SIGALRM, _impatient)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["arc"] for l in SCENES.open() if l.strip()} \
        if SCENES.exists() else set()
    todo = [x for x in arcs(a.limit) if x[0] not in done]
    print(f"{len(done)} arcs done, {len(todo)} to go", flush=True)
    api = _kit_api()
    rows_made = scenes_ok = 0
    for rid, req, beats in todo:
        plan = "\n".join(f"{b.n}. {b.intent}"
                         + (f" -- {b.narration[:240]}" if b.narration else "")
                         for b in beats)
        signal.alarm(300)
        try:
            reply = teacher.ask(TEACHER.format(api=api, request=req[:1200],
                                               plan=plan), max_tokens=6000,
                                system="You write Manim animation code. "
                                       "Code blocks only, in the format asked.")
        except Exception as exc:                              # noqa: BLE001
            signal.alarm(0)
            print(f"  {rid}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            for _ in range(4 if "429" in str(exc) else 1):
                time.sleep(15)
            continue
        signal.alarm(0)
        bodies = bodies_from(reply, len(beats))
        missing = missing_names(beats, bodies, kit=True)
        if missing:
            bodies, _ = prune_statements(bodies, missing)
        asm = assemble(beats, bodies, kit=True)
        res = h.render(asm.code, quality="low", frames=2) if asm.ok else None
        tries = 0
        while res is not None and not res.ok and tries < 3:
            k = failing_beat(asm.code, res.stderr or "")
            if k is None or not bodies[k - 1].strip():
                break
            bodies[k - 1] = ""
            asm = assemble(beats, bodies, kit=True)
            res = h.render(asm.code, quality="low", frames=2) if asm.ok else None
            tries += 1
        ok = bool(res and res.ok)
        kept = [j for j, b in enumerate(bodies) if b.strip() and visual(b, mob)]
        with SCENES.open("a") as f:
            f.write(json.dumps({"arc": rid, "ok": ok, "beats": len(beats),
                                "visual": len(kept),
                                "problems": asm.problems[:2],
                                "error": (res.error_kind.value if res else ""),
                                "stderr": ((res.stderr or "")[-400:]
                                           if res and not res.ok else "")})
                    + "\n")
        if ok and len(kept) * 2 >= len(beats):
            scenes_ok += 1
            with OUT.open("a") as f:
                for j in kept:
                    f.write(json.dumps({
                        "messages": [
                            {"role": "system", "content": CODE_SYSTEM_KIT},
                            {"role": "user",
                             "content": beat_prompt(req, beats, j, bodies)},
                            {"role": "assistant",
                             "content": f"```python\n{bodies[j]}\n```"}],
                        "meta": {"id": f"kit:{rid}:{j}", "scene": f"kit:{rid}",
                                 "source": "kit-teacher", "task": "beat",
                                 "index": j,
                                 "teacher": f"{a.provider}:{a.model}"}},
                        ensure_ascii=False) + "\n")
                    rows_made += 1
        print(f"  {rid}: ok={ok} visual={len(kept)}/{len(beats)} "
              f"(scenes kept {scenes_ok}, rows {rows_made})", flush=True)
        time.sleep(a.pause)
    return 0


if __name__ == "__main__":
    sys.exit(main())
