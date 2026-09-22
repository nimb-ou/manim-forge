#!/usr/bin/env python3
"""Split verified corpus scenes into beats, to train the coder at scale.

    ./.venv/bin/python scripts/decompose_corpus.py --provider mistral

`build_coder_mix.py` gets 182 beat rows out of the gold scenes, which is not
enough to train an adapter. The 1,516 render-verified corpus scenes are the
only other source, and they are written as one `construct` rather than as
beats.

**The split is structural, and only the labels come from a teacher.** Asking
a model to rewrite a scene into beat methods would produce code that has
never rendered, throwing away the one property that makes this corpus worth
anything. So the code is cut verbatim at its own `self.wait(...)` calls --
which is where a beat ends by construction, and which 81% of verified scenes
contain -- and the teacher is asked only to name what each group does.

A wrong label costs a slightly misleading training prompt. A rewritten scene
would cost the render guarantee, silently.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = ("You label the steps of a mathematical animation. Terse, visual, "
          "no commentary.")
PROMPT = (
    "A Manim scene was written to satisfy this request:\n{request}\n\n"
    "Its body is split into the numbered groups below. For each group, write "
    "what the viewer sees happen — the visual step, eight words or fewer.\n\n"
    "Answer with exactly one line per group, numbered the same way, nothing "
    "else.\n\n{body}"
)


def strip_fence(code: str) -> str:
    """The verified index stores code inside a markdown fence.

    Without this every one of 300 sampled scenes raised SyntaxError and the
    decomposer reported them as "skipped" -- a silent zero that looked like
    a corpus with no decomposable scenes in it rather than a two-line bug.
    """
    if "```" not in code:
        return code
    body = max(code.split("```"), key=len)
    return body[len("python"):] if body.startswith("python") else body


def construct_body(src: str) -> tuple[list[ast.stmt], str] | None:
    src = strip_fence(src)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "construct":
            return node.body, src
    return None


def groups(body: list[ast.stmt], src: str) -> list[list[ast.stmt]]:
    """Cut after each `self.wait(...)`, which is where a beat ends.

    A trailing group with no play call is dropped: it is cleanup, not a beat.
    """
    out, cur = [], []
    for stmt in body:
        cur.append(stmt)
        call = stmt.value if isinstance(stmt, ast.Expr) else None
        if isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "wait":
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    keep = []
    for g in out:
        has_play = any(
            isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "play"
            for s in g for n in ast.walk(s))
        if has_play:
            keep.append(g)
    return keep


def text_of(stmts: list[ast.stmt], src: str) -> str:
    segs = [ast.get_source_segment(src, s) or "" for s in stmts]
    return "\n".join(s for s in segs if s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="mistral",
                    help="kept off gemini by default: the planner build is "
                         "already saturating that key")
    ap.add_argument("--model", default="mistral-large-latest")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-groups", type=int, default=3,
                    help="skip scenes too short to be worth decomposing")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "planner" / "coder_corpus.jsonl")
    ap.add_argument("--pause", type=float, default=1.0)
    a = ap.parse_args()

    from forge.synth.teacher import Teacher
    teacher = Teacher(provider=a.provider, model=a.model)

    rows = [json.loads(l) for l in
            (ROOT / "data" / "verified" / "example_index.jsonl").read_text()
            .splitlines() if l.strip()]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if a.out.exists():
        for line in a.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["meta"]["scene"])
        print(f"resuming: {len(done)} scenes already decomposed")

    SYSTEM_CODER = (ROOT / "scripts" / "build_coder_mix.py").read_text()
    m = re.search(r'^SYSTEM = \((.*?)\)\n', SYSTEM_CODER, re.S | re.M)
    coder_system = "".join(re.findall(r'"([^"]*)"', m.group(1))) if m else ""

    made = skipped = failed = 0
    todo = rows[: a.limit or None]
    for n, r in enumerate(todo, 1):
        sid = f"{r.get('source','?')}:{n:06d}"
        if sid in done:
            continue
        try:
            got = construct_body(r["code"])
        except SyntaxError:
            skipped += 1
            continue
        if got is None:
            skipped += 1
            continue
        body, src = got
        gs = groups(body, src)
        if len(gs) < a.min_groups:
            skipped += 1
            continue
        listing = "\n\n".join(
            f"GROUP {i}:\n{text_of(g, src)}" for i, g in enumerate(gs, 1))
        try:
            reply = teacher.ask(PROMPT.format(request=r["prompt"].strip(),
                                              body=listing),
                                max_tokens=600, system=SYSTEM)
        except Exception as exc:                              # noqa: BLE001
            failed += 1
            print(f"  [{n}/{len(todo)}] {type(exc).__name__}", flush=True)
            time.sleep(a.pause * 3)
            continue
        intents = {}
        for line in reply.splitlines():
            # The prompt asks for "1. text" and mistral-medium answers
            # "GROUP 1: text", which the strict pattern rejected -- 0 labels
            # parsed from a perfectly good reply, reported as the scene being
            # skipped. Accept the numbering the model actually uses rather
            # than the one the prompt asked for.
            mm = re.match(r"\s*(?:GROUP|STEP|BEAT)?\s*(\d+)\s*[.):\-]\s*(.+)",
                          line, re.I)
            if mm:
                intents[int(mm.group(1))] = mm.group(2).strip()
        if len(intents) < len(gs):
            failed += 1
            print(f"  [{n}/{len(todo)}] {len(intents)}/{len(gs)} labels, "
                  f"skipped", flush=True)
            time.sleep(a.pause)
            continue

        with a.out.open("a") as f:
            for i, g in enumerate(gs, 1):
                prior = "\n".join(f"  {j}. {intents[j]}" for j in range(1, i)) \
                    or "  (nothing yet — this is the opening beat)"
                user = (f"REQUEST\n{r['prompt'].strip()}\n\n"
                        f"ALREADY ON SCREEN\n{prior}\n\n"
                        f"HELPERS THIS SCENE DEFINES\n  (none)\n\n"
                        f"WRITE THIS BEAT — step {i} of {len(gs)}\n"
                        f"  intent: {intents[i]}")
                f.write(json.dumps({
                    "messages": [{"role": "system", "content": coder_system},
                                 {"role": "user", "content": user},
                                 {"role": "assistant",
                                  "content": f"```python\n{text_of(g, src)}\n```"}],
                    "meta": {"id": f"coder-corpus:{sid}:{i}", "scene": sid,
                             "source": r.get("source", "?"), "task": "beat",
                             "index": i - 1,
                             "loc": len(text_of(g, src).splitlines())},
                }) + "\n")
        made += len(gs)
        done.add(sid)
        if n % 10 == 0 or n < 5:
            print(f"  [{n}/{len(todo)}] {made} beats, {skipped} skipped, "
                  f"{failed} failed", flush=True)
        time.sleep(a.pause)

    print(f"\n{made} beat rows written this pass "
          f"({skipped} scenes skipped, {failed} failed) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
