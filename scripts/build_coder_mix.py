#!/usr/bin/env python3
"""Training rows for the *coder*: one beat in, one method out.

    ./.venv/bin/python scripts/build_coder_mix.py

The other half of the split. `build_planner_mix.py` teaches the arc; this
teaches implementing one step of it.

**Why this shape.** Run 17's failures are 80% longer than its own passes --
23 lines against 13, 19 calls against 11 -- and both scaffolding probes that
tried to fix that from outside the weights did nothing (a pre-render API
check catches 1 of 23; a worked example in the repair prompt changed 77% to
77%). The model is not short of examples or of names. It commits to an
approach too large to execute and then cannot climb out.

So make the unit narrower. The gold scenes are already written in beats --
182 `@beat(...)` methods across 44 scenes.

**And the obvious version of that argument does not survive the data.** The
methods are a median of 28 lines, p90 41, max 55 -- *longer* than the 23-line
failures, not shorter. "One beat is small enough for it" is wrong as stated.

What is actually narrower is the task, not the output. A beat method is
handed its setup: the objects earlier beats left on `self`, the helpers the
scene defines, one stated intent and one duration. Run 17's failures were
23 lines that had to invent the whole scene -- choose the construction,
build it, animate it, and land the point -- from a single sentence. Same
line count, far more decisions per line, and the decisions are where it
reached for a hand-built `Polyhedron`.

That is the claim this data can support, and it is weaker than the one I
started with. Whether it holds is what training it answers.

**What each row carries.** A beat is not independent: beat 3 moves what beat
1 put on screen. So the prompt includes the request, the beats already
played, and the helpers the scene defines, and the answer is that one
method. Without the running context the target is unlearnable -- the method
references `self.arrows` and there would be nothing saying where it came
from.

**One output format, checked.** The gold rows first emitted the whole
`def name(self): ...` method while the decomposed corpus rows emitted bare
statements -- two formats for one task, which would have taught the model to
produce either and broken assembly on whichever it did not expect. That cost
nothing to find here and would have cost a five-hour GPU run to find at eval.
Both now emit the dedented body, which is also what the harness wants to
concatenate.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import textwrap
from pathlib import Path

from forge.app.twostage import names_in_scope

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (
    "You write one beat of a 3Blue1Brown-style Manim scene. You are given the "
    "whole request, the beats already on screen, and the helpers the scene "
    "defines. Write only the code for the beat you are asked for, as "
    "statements at method-body level -- no class, no def, no imports. "
    "NAMES IN SCOPE lists what earlier beats already built: reuse those "
    "rather than rebuilding them. Anything else you use you must "
    "CONSTRUCT in this beat before you animate it -- `self.play(Create(dot))` "
    "is wrong unless a line above it makes `dot`."
)


def beat_methods(src: str) -> list[dict]:
    """Decorated beat methods with their spec, in source order."""
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.decorator_list:
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            fname = dec.func.id if isinstance(dec.func, ast.Name) else \
                getattr(dec.func, "attr", "")
            if fname != "beat":
                continue
            kw = {k.arg: k.value for k in dec.keywords}
            def const(v):
                return v.value if isinstance(v, ast.Constant) else None
            body = "\n".join(
                textwrap.dedent(ast.get_source_segment(src, st) or "")
                for st in node.body)
            out.append({
                "name": node.name,
                "body": body,
                "intent": const(dec.args[0]) if dec.args else "",
                "seconds": const(kw.get("seconds")) if "seconds" in kw else None,
                "narration": " ".join(str(const(kw.get("narration")) or "").split()),
                "source": ast.get_source_segment(src, node) or "",
                "lineno": node.lineno,
            })
    return sorted(out, key=lambda b: b["lineno"])


def localise(src: str, attrs: set[str]) -> str:
    """Rewrite `self.x` to `x` for attributes the scene assigns.

    The gold scenes pass state between beats on `self`, because @beat methods
    have no other way to talk. The 8,649 decomposed corpus rows use plain
    locals, because they were cut out of one construct where locals already
    persist. That is two conventions for one task, and the coder learned the
    minority one -- gold is weighted six times -- then read `self.axes` in
    beat one and every assembled scene died on api_misuse.

    Only attributes the scene *assigns* are rewritten, which leaves
    `self.play`, `self.wait`, `self.add` and the scene's helper methods
    alone: those are Scene's, not the beat's.
    """
    if not attrs:
        return src
    pattern = re.compile(r"\bself\.(" + "|".join(
        re.escape(a) for a in sorted(attrs, key=len, reverse=True)) + r")\b")
    return pattern.sub(r"\1", src)


def assigned_attrs(src: str) -> set[str]:
    """Attributes the scene assigns to self, anywhere."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    return {n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
            and isinstance(n.value, ast.Name) and n.value.id == "self"}


def helpers(src: str) -> str:
    """Undecorated methods and module-level defs the beats may call.

    Signatures only. The bodies would triple the prompt and the coder needs
    to know a helper *exists and what it takes*, not how it is implemented.
    """
    tree = ast.parse(src)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and not node.decorator_list:
            if node.name.startswith("__"):
                continue
            args = [a.arg for a in node.args.args if a.arg != "self"]
            names.append(f"  {node.name}({', '.join(args)})")
    return "\n".join(dict.fromkeys(names))


def module_constants(src: str) -> list[tuple[set[str], str]]:
    """Module-level assignments, in order, as (names bound, source line)."""
    tree = ast.parse(src)
    out = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) \
                else [node.target]
            names = {t.id for tg in targets for t in ast.walk(tg)
                     if isinstance(t, ast.Name)}
            if names:
                out.append((names, ast.get_source_segment(src, node) or ""))
        # Module-level functions and small classes (`f`, `relu`, `Vec2`) are
        # the same problem one level up: defined outside the beat, read in
        # it. The Scene subclass itself is not a dependency, it is the scene.
        elif isinstance(node, ast.FunctionDef) and node.name != "beat":
            out.append(({node.name}, ast.get_source_segment(src, node) or ""))
        elif isinstance(node, ast.ClassDef) and not any(
                "Scene" in ast.unparse(b) for b in node.bases):
            out.append(({node.name}, ast.get_source_segment(src, node) or ""))
    return out


def with_constants(body: str, consts: list[tuple[set[str], str]]) -> str:
    """Prepend the module constants the beat reads, and what they read.

    Gold scenes keep colours and parameters at module level -- `DIM`,
    `SUM_C`, `K` -- and the beats read them. The assembled scene has no
    module level, so a coder that learns to read `DIM` writes a name nothing
    defines. Prepending the definitions makes each row obey the CONSTRUCT
    rule it is trained under: build what is not in scope, then use it.
    """
    def loads(src: str) -> set[str]:
        try:
            return {n.id for n in ast.walk(ast.parse(src))
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        except SyntaxError:
            return set()
    need, picked = loads(body), set()
    changed = True
    while changed:
        changed = False
        for i, (names, line) in enumerate(consts):
            if i not in picked and names & need:
                picked.add(i)
                need |= loads(line)
                changed = True
    if not picked:
        return body
    return "\n".join(consts[i][1] for i in sorted(picked)) + "\n" + body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "planner" / "coder.jsonl")
    a = ap.parse_args()

    rows, skipped = [], 0
    for line in (ROOT / "data" / "gold" / "gold.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        beats = beat_methods(g["code"])
        if not beats:
            skipped += 1
            continue
        attrs = assigned_attrs(g["code"])
        consts = module_constants(g["code"])
        for b in beats:
            b["body"] = with_constants(localise(b["body"], attrs), consts)
        # The runtime has no helpers, and the beats kept below call none, so
        # the prompt says so rather than listing methods the answer ignores.
        helps = ""
        # Beats that call the scene's own helpers cannot be training data
        # for a runtime that has none. The two-stage harness assembles bare
        # `Scene` subclasses, so `self.panel(...)` is a method that will
        # never exist -- and 139 of the 182 gold beats call one. At weight
        # six they taught the coder to reach for infrastructure the runtime
        # cannot supply, which is a large part of why every assembled scene
        # died on api_misuse.
        #
        # 43 rows survive, across 22 of the 44 scenes. Small, and the only
        # part of gold that transfers. The rest are not deleted -- the
        # scenes still serve the animation gate and the planner.
        import manim as _manim
        _scene_api = set(dir(_manim.Scene))
        usable = []
        for b in beats:
            calls = {n for n in re.findall(r"self\.(\w+)", b["body"])
                     if n not in _scene_api}
            if not calls:
                usable.append(b)
        if not usable:
            skipped += 1
            continue
        for i, b in enumerate(beats):
            if b not in usable:
                continue
            prior = "\n".join(
                f"  {j + 1}. {p['intent']}" for j, p in enumerate(beats[:i])
            ) or "  (nothing yet — this is the opening beat)"
            secs = f"{b['seconds']:g} seconds" if b["seconds"] else "a few seconds"
            scope = names_in_scope([p["body"] for p in beats[:i]])
            user = (
                f"REQUEST\n{g['prompt'].strip()}\n\n"
                f"ALREADY ON SCREEN\n{prior}\n\n"
                f"NAMES IN SCOPE\n  "
                + (", ".join(scope) if scope else "(none yet)") + "\n\n"
                f"HELPERS THIS SCENE DEFINES\n{helps or '  (none)'}\n\n"
                f"WRITE THIS BEAT — step {i + 1} of {len(beats)}, "
                f"about {secs}\n"
                f"  intent: {b['intent']}\n"
                f"  narration: {b['narration']}"
            )
            rows.append({
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": user},
                             {"role": "assistant",
                              "content": f"```python\n{b['body']}\n```"}],
                "meta": {"id": f"coder-gold:{g['meta']['scene']}:{b['name']}",
                         "source": "gold", "task": "beat",
                         "scene": g["meta"]["scene"], "index": i,
                         "loc": len(b["body"].splitlines())},
            })

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    locs = sorted(r["meta"]["loc"] for r in rows)
    print(f"{len(rows)} beat rows from {44 - skipped} scenes -> {a.out}")
    if locs:
        print(f"method length: median {locs[len(locs)//2]} lines, "
              f"p10 {locs[len(locs)//10]}, p90 {locs[9*len(locs)//10]}, "
              f"max {locs[-1]}")
        print(f"  (run 17's passes averaged 13 lines, its failures 23)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
