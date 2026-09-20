"""Rewrite each beat's declared ``seconds`` to a duration its narration fits.

``seconds=`` was written by eye before the narration existed, and the audit
shows nearly every beat under-declaring by roughly half. That matters twice
over: the rendered animation runs out before the voice track does, and -- far
worse -- these scenes are the corpus's style anchor, so a model trained on
them learns to pair eight seconds of picture with twenty seconds of script.

Edits are made through the AST, by column offset of the ``seconds`` keyword
node, rather than by regex over the file. ``seconds=8`` appears in more places
than the decorator.
"""
from __future__ import annotations

import argparse
import ast
import io
from pathlib import Path

WPS = 2.6
MIN_S = 3.0


def recommend(narration: str, declared: float) -> int:
    words = len(narration.split())
    needed = words / WPS if words else declared
    return max(MIN_S, int(needed) + (1 if needed > int(needed) else 0),
               int(declared))


def retime(path: Path, apply: bool) -> list[tuple[str, float, int]]:
    src = io.open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    edits, report = [], []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if not (isinstance(dec, ast.Call)
                    and getattr(dec.func, "id", None) == "beat"):
                continue
            sec_kw = narr_kw = None
            for kw in dec.keywords:
                if kw.arg == "seconds":
                    sec_kw = kw
                elif kw.arg == "narration":
                    narr_kw = kw
            if sec_kw is None or narr_kw is None:
                continue
            try:
                declared = float(ast.literal_eval(sec_kw.value))
                narration = ast.literal_eval(narr_kw.value)
            except (ValueError, TypeError):
                continue
            want = recommend(narration, declared)
            if want == declared:
                continue
            report.append((node.name, declared, want))
            edits.append((sec_kw.value.lineno - 1,
                          sec_kw.value.col_offset,
                          sec_kw.value.end_col_offset,
                          str(want)))

    if apply and edits:
        # Right to left within each line, so earlier offsets stay valid.
        for ln, c0, c1, text in sorted(edits, key=lambda e: (-e[0], -e[1])):
            line = lines[ln]
            lines[ln] = line[:c0] + text + line[c1:]
        io.open(path, "w", encoding="utf-8").write("".join(lines))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    total = 0
    for p in sorted(Path("forge/gold").glob("*.py")):
        if p.name in ("__init__.py", "curriculum.py"):
            continue
        rep = retime(p, a.apply)
        if not rep:
            continue
        print(f"\n{p.name}")
        for name, was, now in rep:
            print(f"   {name:<22} {was:>5.0f}s -> {now:>3}s")
            total += 1
    verb = "retimed" if a.apply else "would retime"
    print(f"\n{verb} {total} beats" + ("" if a.apply else "   (--apply to write)"))


if __name__ == "__main__":
    main()
