"""Hand-written kit scenes as training rows: the blocks no teacher has used.

The six classic-picture blocks (squares_on_sides, angle_sum, count_binary,
secant_to_tangent, swing_pendulum, sieve_primes) were added after the
teacher's budget ran out, so no training row calls them, and the kit
coder under-uses the motion blocks it does know (gradient_descent,
riemann_refine, secant slopes): on "gradient descent" it drew small arrows
beside a parabola. These scenes, written by hand in the teacher's format,
show each block on requests phrased differently from the evaluation's.
Each scene is rendered whole; only scenes that render become rows, one per
beat, with the scene's earlier beats as the prefix -- exactly the
teacher's rows (synth_kit_beats.py) -- tagged source "kit-claude".

    ./.venv/bin/python scripts/claude_kit_scenes.py
      -> data/kit/claude_beats.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forge.app.pipeline import CODE_SYSTEM_KIT_TRAINED  # noqa: E402
from forge.app.twostage import Beat, assemble, beat_prompt  # noqa: E402

OUT = ROOT / "data" / "kit" / "claude_beats.jsonl"

from forge.kit.exemplars import SCENES  # noqa: E402,F401

def main() -> int:
    from forge.harness import RenderHarness
    h = RenderHarness(python_bin=str(ROOT / ".venv" / "bin" / "python"),
                      cache_dir=str(ROOT / "data" / "frames"), timeout=600)
    rows, bad = [], 0
    for sid, (req, spec) in enumerate(SCENES):
        beats = [Beat(k + 1, None, i, n) for k, (i, n, _) in enumerate(spec)]
        bodies = [c for _, _, c in spec]
        asm = assemble(beats, bodies, kit=True)
        res = h.render(asm.code, quality="low", frames=2) if asm.ok else None
        if not (res and res.ok):
            bad += 1
            why = asm.problems if not asm.ok else (res.stderr or "")[-400:]
            print(f"  FAIL {sid}: {req[:50]} -- {why}", flush=True)
            continue
        for j in range(len(beats)):
            rows.append({
                "messages": [
                    {"role": "system", "content": CODE_SYSTEM_KIT_TRAINED},
                    {"role": "user", "content": beat_prompt(req, beats, j, bodies)},
                    {"role": "assistant", "content": f"```python\n{bodies[j]}\n```"}],
                "meta": {"id": f"claude:{sid}:{j}", "scene": f"claude:{sid}",
                         "source": "kit-claude", "task": "beat", "index": j},
                "prefix": bodies[:j],
                "intents": [b.intent for b in beats[: j + 1]],
                "request": req})
        print(f"  ok   {sid}: {req[:60]} ({len(beats)} beats)", flush=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} rows from {len(SCENES) - bad}/{len(SCENES)} scenes -> {OUT}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
