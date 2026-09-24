"""GRPO prompts: a beat to write, on top of the coder's *own* earlier beats.

The coder renders 92.7% of beats when the beats before it are the reference
bodies, and far fewer scenes end to end, because at inference the beats
before it are its own. So the prompts come from real two-stage runs: every
assembled scene in data/bench is split at its `# beat N:` comments, and for
each beat j the prompt is the inference prompt (beat_prompt, the one
definition) over the model-written bodies of beats < j.

A prompt is kept only if its prefix renders on its own -- then a failure
after adding the completion is the completion's. The reward itself runs in
the kernel: assemble prefix + completion, render, score.

    ./.venv/bin/python -u scripts/build_grpo_prompts.py
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import shutil
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from forge.app.twostage import Beat, assemble, beat_prompt  # noqa: E402
from run_twostage import CODE_SYSTEM  # noqa: E402

OUT = ROOT / "kaggle" / "manim-forge-grpo"
BEAT = re.compile(r"^        # beat (\d+): (.*)$", re.M)


def split_scene(code: str) -> tuple[list[Beat], list[str]]:
    """Beats and their bodies back out of an assembled scene."""
    marks = list(BEAT.finditer(code))
    beats, bodies = [], []
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(code)
        body = textwrap.dedent(code[m.end():end]).strip("\n")
        body = re.sub(r"\n?self\.wait\(1\)\s*$", "", body) \
            if k + 1 == len(marks) else body
        beats.append(Beat(n=len(beats) + 1, seconds=None,
                          intent=m.group(2).strip()))
        bodies.append(body)
    return beats, bodies


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-per-scene", type=int, default=8)
    a = ap.parse_args()

    from forge.harness import RenderHarness
    h = RenderHarness(python_bin=str(ROOT / ".venv" / "bin" / "python"),
                      cache_dir=str(ROOT / "data" / "frames"), timeout=120)

    seen, rows = set(), []
    files = sorted(glob.glob(str(ROOT / "data" / "bench" / "*.json")))
    for f in files:
        try:
            d = json.loads(Path(f).read_text())
        except Exception:                                     # noqa: BLE001
            continue
        for t in (d.get("trials") or []) if isinstance(d, dict) else []:
            code, req = t.get("code") or "", t.get("request") or ""
            if "# beat" not in code or not req:
                continue
            beats, bodies = split_scene(code)
            # Blanked (salvaged) beats have no body to learn from.
            live = [j for j, b in enumerate(bodies) if b.strip()]
            for j in live[: a.max_per_scene]:
                prefix = bodies[:j]
                key = hashlib.sha256((req + "\0".join(prefix) + beats[j].intent)
                                     .encode()).hexdigest()[:16]
                if key in seen:
                    continue
                seen.add(key)
                if any(p.strip() for p in prefix):
                    asm = assemble(beats[:j], prefix)
                    if not asm.ok or not h.render(asm.code, quality="low",
                                                  frames=2).ok:
                        continue
                rows.append({
                    "prompt": [{"role": "system", "content": CODE_SYSTEM},
                               {"role": "user", "content":
                                beat_prompt(req, beats, j, bodies)}],
                    "prefix": prefix,
                    "intents": [b.intent for b in beats[: j + 1]],
                    "request": req, "j": j, "id": key,
                    "source": Path(f).stem,
                })
        print(f"  {Path(f).stem}: {len(rows)} prompts so far", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prompts.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    # The kernel scores with the same assemble() inference uses.
    shutil.copy2(ROOT / "forge" / "app" / "twostage.py", OUT / "twostage.py")
    print(f"{len(rows)} prompts -> {OUT / 'prompts.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
