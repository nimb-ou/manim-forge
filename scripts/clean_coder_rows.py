"""Make the coder's training rows say one thing, and agree with inference.

Coder v2 trained on 8,649 corpus rows whose system prompt said "write only
the method ... its signature, its body", on 43 gold rows with a second
prompt, and was then asked at inference with a third. The CONSTRUCT rule --
anything not in NAMES IN SCOPE is built in this beat before it is animated --
was in none of the prompts it trained on, and some rows break it: the
reference answer animates an object no earlier beat built and this beat does
not build either. Training on those teaches exactly the cross-beat failure
the split renders keep hitting.

This rewrites every row's system prompt to the one run_twostage.py sends and
drops rows whose answer uses a name that is neither in scope, built in the
beat, a Manim export, a builtin, nor a helper the scene defines. The
originals go to data/planner/attic/ first.

    ./.venv/bin/python scripts/clean_coder_rows.py            # report only
    ./.venv/bin/python scripts/clean_coder_rows.py --write
"""
from __future__ import annotations

import argparse
import ast
import builtins
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from forge.app.twostage import defined_names, extract_code  # noqa: E402
from build_coder_mix import SYSTEM  # noqa: E402

FILES = [ROOT / "data" / "planner" / "coder.jsonl",
         ROOT / "data" / "planner" / "coder_corpus.jsonl"]


def section(user: str, head: str) -> list[str]:
    m = re.search(rf"^{head}\n(.*?)(?:\n\n|\Z)", user, re.S | re.M)
    if not m:
        return []
    text = m.group(1).strip()
    if text == "(none)":
        return []
    return [t.strip() for t in re.split(r"[,\n]", text) if t.strip()]


def undefined(row: dict, known: set[str], scene_attrs: set[str]) -> list[str]:
    """Names the answer uses that nothing gives it, or ['<unparsable>']."""
    user = row["messages"][1]["content"]
    code = extract_code(row["messages"][-1]["content"])
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["<unparsable>"]
    scope = set(section(user, "NAMES IN SCOPE"))
    helpers = {h.split("(")[0].strip() for h in
               section(user, "HELPERS THIS SCENE DEFINES")}
    plain = {s for s in scope if not s.startswith("self.")}
    attrs = {s[5:] for s in scope if s.startswith("self.")} | plain | helpers
    # Lambda and def parameters are ast.arg, not Name, so defined_names
    # misses them: `lambda x: x**2` would read as an undefined `x`.
    args = {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    have = known | plain | helpers | args | defined_names(tree) | {"self"}
    set_attrs = {n.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute)
                 and isinstance(n.ctx, ast.Store)
                 and isinstance(n.value, ast.Name) and n.value.id == "self"}
    out = {n.id for n in ast.walk(tree)
           if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
           and n.id not in have}
    out |= {f"self.{n.attr}" for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)
            and isinstance(n.value, ast.Name) and n.value.id == "self"
            and n.attr not in set_attrs | attrs | scene_attrs}
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    import manim
    # np and math come from the preamble every assembled scene starts with.
    known = ({n for n in dir(manim) if not n.startswith("_")}
             | set(dir(builtins)) | {"np", "math"})
    # Instance attributes Scene sets in __init__, which dir(Scene) misses.
    scene_attrs = set(dir(manim.Scene)) | {"mobjects", "camera", "renderer",
                                           "foreground_mobjects", "time"}

    attic = ROOT / "data" / "planner" / "attic"
    for path in FILES:
        rows = [json.loads(l) for l in path.open() if l.strip()]
        keep, why = [], Counter()
        prompts = Counter(r["messages"][0]["content"] for r in rows)
        for r in rows:
            bad = undefined(r, known, scene_attrs)
            if bad:
                why["unparsable" if bad == ["<unparsable>"] else "undefined"] += 1
                for b in bad:
                    why[f"  {b}"] += 1
                continue
            r["messages"][0]["content"] = SYSTEM
            keep.append(r)
        print(f"{path.name}: {len(rows)} rows, {len(prompts)} system prompt(s)"
              f" -> {len(keep)} kept, dropped {len(rows) - len(keep)} "
              f"({why['undefined']} undefined names, {why['unparsable']} "
              f"unparsable)")
        top = [(k.strip(), v) for k, v in why.most_common(40)
               if k.startswith("  ")][:12]
        print("  most common missing:", ", ".join(f"{k}×{v}" for k, v in top))
        if a.write:
            attic.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, attic / f"{path.stem}.pre-v3.jsonl")
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                    for r in keep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
