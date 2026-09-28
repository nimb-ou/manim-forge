"""Render hand-written kit scenes and say, per scene, whether they work.

For the Claude-written teacher batches (data/kit/claude_scenes/*.json): each
scene is assembled exactly as the pipeline assembles a plan's beats (the
kit embedded, `stage = Stage(self)`), rendered at low quality with a mark
at the end of each beat, and a contact sheet of the beat-end frames is
written beside the batch so the writer can look at what each beat shows.

    ./.venv/bin/python scripts/check_scenes.py data/kit/claude_scenes/batch_03.json
      -> prints "OK k" / "FAIL k: <error> at <line>" per scene
      -> data/kit/claude_scenes/batch_03_sheets/<k>.jpg

A batch file is a JSON list of scenes:
  {"request": "...", "beats": [{"intent": "...", "narration": "...",
                                "code": "..."}, ...]}
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from forge.app.twostage import (Beat, assemble, error_message,  # noqa: E402
                                failing_beat, failing_line)


def main() -> int:
    path = Path(sys.argv[1])
    only = {int(x) for x in sys.argv[2].split(",")} if len(sys.argv) > 2 else None
    scenes = json.loads(path.read_text())
    from forge.harness import RenderHarness
    from scorecard import contact_sheet
    h = RenderHarness(python_bin=str(ROOT / ".venv" / "bin" / "python"),
                      cache_dir=str(ROOT / "data" / "frames"), timeout=300,
                      store_video=True)
    sheets = path.with_name(path.stem + "_sheets")
    sheets.mkdir(exist_ok=True)
    ok_n = 0
    for k, sc in enumerate(scenes):
        if only is not None and k not in only:
            continue
        beats = [Beat(j + 1, None, b["intent"], b.get("narration", ""))
                 for j, b in enumerate(sc["beats"])]
        bodies = [b["code"].strip() + "\nstage.mark()" for b in sc["beats"]]
        asm = assemble(beats, bodies, kit=True)
        if not asm.ok:
            print(f"FAIL {k}: does not assemble: {'; '.join(asm.problems)[:300]}", flush=True)
            continue
        res = h.render(asm.code, quality="low", use_cache=False)
        if not res.ok:
            n = failing_line(asm.code, res.stderr or "")
            b = failing_beat(asm.code, res.stderr or "")
            line = asm.code.splitlines()[n - 1].strip() if n else "?"
            print(f"FAIL {k}: beat {b}: {error_message(res.stderr or '') or res.error_kind.value}"
                  f" at `{line[:100]}`", flush=True)
            continue
        got = re.findall(r"BEAT_ENDS \[([^\]]*)\]", res.stderr or "")
        ends = [float(x) for x in got[-1].split(",")] if got else []
        contact_sheet(res.video_path, sheets / f"{k:02d}.jpg", times=ends)
        ok_n += 1
        print(f"OK {k}  ({len(beats)} beats, {res.duration_s or 0:.0f}s) sheet: {sheets / f'{k:02d}.jpg'}",
              flush=True)
    print(f"{ok_n} rendered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
