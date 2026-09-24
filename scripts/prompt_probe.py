"""Does one sentence in the coder's system prompt stop text-only beats?

On the hard eval 215 of 289 beats built no non-text mobject, for intents
like "Light exiting the glass" and "Basis vectors i-hat and j-hat
highlighted" -- the coder answers visual intents with captions. Retraining
is queued; this is the free test first. The same beats (the coder's own
earlier beats as prefix, via build_grpo_prompts' prompts) are generated
under the trained system prompt and under it plus one sentence, and each
beat is checked for a non-text mobject, motion, parsing and rendering.

    ./.venv/bin/python -u scripts/prompt_probe.py --n 40
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from forge.app.twostage import Beat, assemble, extract_code  # noqa: E402
from run_twostage import CODE_SYSTEM, ask, load  # noqa: E402

EXTRA = (" Draw what the intent names -- shapes, arrows, axes, graphs, "
         "diagrams -- and use text only to label them.")
TEXT = {"Text", "MathTex", "Tex", "Title", "MarkupText", "Paragraph",
        "BulletedList", "Code", "Integer", "DecimalNumber", "Variable"}


def shape_motion(code: str, mob: set[str]) -> tuple[bool, bool, bool]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, False, False
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return (True, bool((calls & mob) - TEXT) or ".plot(" in code,
            ".animate" in code or bool(calls & {"Transform", "Rotate",
                                                "ReplacementTransform",
                                                "MoveAlongPath"}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--coder", default=str(ROOT / "adapters" / "mlx-coder2"))
    a = ap.parse_args()

    import manim
    mob = {n for n in dir(manim) if isinstance(getattr(manim, n), type)
           and issubclass(getattr(manim, n), manim.Mobject)}
    from forge.harness import RenderHarness
    h = RenderHarness(python_bin=str(ROOT / ".venv" / "bin" / "python"),
                      cache_dir=str(ROOT / "data" / "frames"), timeout=120)

    rows = [json.loads(l) for l in
            (ROOT / "kaggle" / "manim-forge-grpo" / "prompts.jsonl").open()]
    rows = rows[:: max(1, len(rows) // a.n)][: a.n]
    model, tok = load(a.coder)
    out = {}
    for name, system in (("trained prompt", CODE_SYSTEM),
                         ("+ draw-it sentence", CODE_SYSTEM + EXTRA)):
        t = {"parse": 0, "shape": 0, "motion": 0, "render": 0}
        for r in rows:
            code = extract_code(ask(model, tok, system,
                                    r["prompt"][1]["content"], 900))
            p, s, m = shape_motion(code, mob)
            t["parse"] += p
            t["shape"] += s
            t["motion"] += m
            beats = [Beat(n=k + 1, seconds=None, intent=x)
                     for k, x in enumerate(r["intents"])]
            asm = assemble(beats, list(r["prefix"]) + [code])
            t["render"] += bool(asm.ok and h.render(asm.code, quality="low",
                                                    frames=2).ok)
        out[name] = t
        print(f"{name:20s} " + "  ".join(f"{k} {v}/{len(rows)}"
                                         for k, v in t.items()), flush=True)
    (ROOT / "data" / "bench" / "prompt_probe.json").write_text(
        json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
