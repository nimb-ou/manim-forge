"""Claude-written kit scenes as training rows, for the coder and the planner.

Two sources, both written by Claude in the teacher's format: the worked
scenes in forge/kit/exemplars.py (the first 31, for the six classic-picture
blocks and the under-used motion blocks) and the subagent batches in
forge/kit/teacher/batch_*.json (docs/TEACHER_GUIDE.md: broad topics,
never the held-out ones, render-checked and looked at by their writers).

Each scene is rendered whole with a mark at every beat's end; only scenes
that render become rows:
  * coder rows, one per beat, the earlier beats as prefix -- exactly the
    teacher's rows -- source "kit-claude" -> data/kit/claude_beats.jsonl;
  * planner rows, the whole arc as one window "N. [seconds] intent --
    narration ... END", seconds from the rendered beat lengths -- source
    "plan-claude" -> data/kit/claude_plans.jsonl.

    ./.venv/bin/python scripts/claude_kit_scenes.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forge.app.pipeline import CODE_SYSTEM_KIT_TRAINED, PLAN_SYSTEM  # noqa: E402
from forge.app.twostage import Beat, assemble, beat_prompt  # noqa: E402
from forge.kit.exemplars import SCENES  # noqa: E402

OUT = ROOT / "data" / "kit" / "claude_beats.jsonl"
PLANS = ROOT / "data" / "kit" / "claude_plans.jsonl"
BATCHES = ROOT / "forge" / "kit" / "teacher"   # tracked; data/ is not


def scenes() -> list[tuple[str, str, list[tuple[str, str, str]]]]:
    """(scene id, request, [(intent, narration, code)]) from both sources."""
    out = [(f"claude:{k}", req, spec) for k, (req, spec) in enumerate(SCENES)]
    for f in sorted(BATCHES.glob("batch_*.json")):
        for k, sc in enumerate(json.loads(f.read_text())):
            out.append((f"claude:{f.stem}:{k}", sc["request"],
                        [(b["intent"], b.get("narration", ""), b["code"])
                         for b in sc["beats"]]))
    return out


def main() -> int:
    from forge.harness import RenderHarness
    h = RenderHarness(python_bin=str(ROOT / ".venv" / "bin" / "python"),
                      cache_dir=str(ROOT / "data" / "frames"), timeout=600)
    rows, plans, bad, todo = [], [], 0, scenes()
    for sid, req, spec in todo:
        beats = [Beat(k + 1, None, i, n) for k, (i, n, _) in enumerate(spec)]
        bodies = [c.strip() for _, _, c in spec]
        asm = assemble(beats, [b + "\nstage.mark()" for b in bodies], kit=True)
        res = h.render(asm.code, quality="low", frames=2) if asm.ok else None
        if not (res and res.ok):
            bad += 1
            why = asm.problems if not asm.ok else (res.stderr or "")[-300:]
            print(f"  FAIL {sid}: {req[:50]} -- {str(why)[:200]}", flush=True)
            continue
        got = re.findall(r"BEAT_ENDS \[([^\]]*)\]", res.stderr or "")
        ends = [float(x) for x in got[-1].split(",")] if got else []
        secs = [max(4, round(b - a)) for a, b in zip([0.0] + ends, ends)] \
            if len(ends) == len(beats) else [None] * len(beats)
        for j in range(len(beats)):
            rows.append({
                "messages": [
                    {"role": "system", "content": CODE_SYSTEM_KIT_TRAINED},
                    {"role": "user", "content": beat_prompt(req, beats, j, bodies)},
                    {"role": "assistant", "content": f"```python\n{bodies[j]}\n```"}],
                "meta": {"id": f"{sid}:{j}", "scene": sid, "source": "kit-claude",
                         "task": "beat", "index": j},
                "prefix": bodies[:j],
                "intents": [b.intent for b in beats[: j + 1]],
                "request": req})
        arc = "\n".join(f"{k + 1}. [{s or 10}s] {i} -- {n}".rstrip(" -")
                        for k, ((i, n, _), s) in enumerate(zip(spec, secs)))
        plans.append({"messages": [
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": f"REQUEST\n{req}\n\nBEATS SO FAR\n"
             "  (nothing yet — open the explanation)\n\nWrite the next 6 beat(s), "
             "numbered from 1. Stop early and write END if the explanation is complete."},
            {"role": "assistant", "content": arc + "\nEND"}],
            "meta": {"id": f"plan-{sid}", "source": "plan-claude", "task": "plan-window",
                     "n_beats": str(len(beats)), "request": req, "final": "True"}})
        print(f"  ok   {sid}: {req[:60]} ({len(beats)} beats)", flush=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    PLANS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in plans))
    print(f"{len(rows)} coder rows, {len(plans)} plans from {len(todo) - bad}/{len(todo)} "
          f"scenes -> {OUT.name}, {PLANS.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
