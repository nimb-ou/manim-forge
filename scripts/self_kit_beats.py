"""The kit coder as its own data source: generate, render, keep what works.

With the teacher's API budget spent, the one abundant resource is this Mac.
Rejection-sampling fine-tuning (STaR, RFT): the kit coder writes every beat
of a drawable arc -- each beat on top of its *own* earlier beats, exactly as
at inference -- the scene is pruned of names nothing builds, rendered, and
a beat the traceback blames is dropped and the rest re-rendered. The beats
that survive, that draw something, and that the relevance and novelty
filters accept become training rows in the teacher's format, so the next
kit coder trains on what this one got right.

Rows go to data/kit/self_beats.jsonl with source "kit-self";
filter_kit_beats.py reads both files. Resumable, one arc at a time.

    ./.venv/bin/python -u scripts/self_kit_beats.py --coder adapters/mlx-coder5-kit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from forge.app.pipeline import CODE_SYSTEM_KIT_TRAINED, load, write_beat  # noqa: E402
from forge.app.twostage import assemble, beat_prompt, failing_beat, prune_all  # noqa: E402
from scorecard import visual  # noqa: E402
from synth_kit_beats import arcs  # noqa: E402

OUT = ROOT / "data" / "kit" / "self_beats.jsonl"
DONE = ROOT / "data" / "kit" / "self_scenes.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coder", default=str(ROOT / "adapters" / "mlx-coder5-kit"))
    ap.add_argument("--beats", type=int, default=8)
    ap.add_argument("--limit", type=int, default=5000)
    a = ap.parse_args()

    import manim
    mob = {n for n in dir(manim) if isinstance(getattr(manim, n), type)
           and issubclass(getattr(manim, n), manim.Mobject)}
    from forge.harness import RenderHarness
    h = RenderHarness(python_bin=str(ROOT / ".venv" / "bin" / "python"),
                      cache_dir=str(ROOT / "data" / "frames"), timeout=300)
    model, tok = load(a.coder)
    done = {json.loads(l)["arc"] for l in DONE.open() if l.strip()} \
        if DONE.exists() else set()
    todo = [x for x in arcs(a.limit) if x[0] not in done]
    print(f"{len(done)} arcs done, {len(todo)} to go", flush=True)
    made = 0
    for rid, req, beats in todo:
        t0 = time.time()
        beats = beats[: a.beats]
        bodies: list[str] = []
        for j in range(len(beats)):
            body, _ = write_beat(model, tok, req, beats, j, bodies, 900,
                                 CODE_SYSTEM_KIT_TRAINED)
            bodies.append(body)
        bodies, _, _ = prune_all(beats, bodies, kit=True)
        asm = assemble(beats, bodies, kit=True)
        res = h.render(asm.code, quality="low", frames=2) if asm.ok else None
        tries = 0
        while res is not None and not res.ok and tries < 3:
            k = failing_beat(asm.code, res.stderr or "")
            if k is None or not bodies[k - 1].strip():
                break
            bodies[k - 1] = ""
            bodies, _, _ = prune_all(beats, bodies, kit=True)
            asm = assemble(beats, bodies, kit=True)
            res = h.render(asm.code, quality="low", frames=2) if asm.ok else None
            tries += 1
        ok = bool(res and res.ok)
        kept = [j for j, b in enumerate(bodies) if b.strip() and visual(b, mob)] \
            if ok else []
        with OUT.open("a") as f:
            for j in kept:
                f.write(json.dumps({
                    "messages": [
                        {"role": "system", "content": CODE_SYSTEM_KIT_TRAINED},
                        {"role": "user",
                         "content": beat_prompt(req, beats, j, bodies)},
                        {"role": "assistant",
                         "content": f"```python\n{bodies[j]}\n```"}],
                    "meta": {"id": f"self:{rid}:{j}", "scene": f"self:{rid}",
                             "source": "kit-self", "task": "beat", "index": j,
                             "coder": Path(a.coder).name},
                    "prefix": bodies[:j],
                    "intents": [b.intent for b in beats[: j + 1]],
                    "request": req}, ensure_ascii=False) + "\n")
        made += len(kept)
        with DONE.open("a") as f:
            f.write(json.dumps({"arc": rid, "ok": ok, "beats": len(beats),
                                "visual": len(kept),
                                "seconds": round(time.time() - t0)}) + "\n")
        print(f"  {rid}: ok={ok} kept={len(kept)}/{len(beats)} "
              f"({time.time() - t0:.0f}s; rows so far {made})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
