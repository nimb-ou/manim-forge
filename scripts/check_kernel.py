#!/usr/bin/env python3
"""Order-aware name check for a Kaggle kernel script.

`ast.parse` proves a file is syntactically Python. It does not prove the file
will survive its own first second, and twice now that gap has cost a GPU
session:

  * a class statement whose *base* was imported further down the file, which
    raises NameError the moment the class body is evaluated; and
  * a patch of mine whose first textual match was inside a comment, leaving a
    name used above where it was bound.

Both are the same defect -- a name read before anything binds it -- and both
are visible statically if you walk the module in source order instead of
asking whether it parses. This does that, and nothing cleverer: it is meant
to be trusted about the failures above, not to be a type checker.

    python3 scripts/check_kernel.py kaggle/01_sft.py
"""
from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path

BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}


def _bound_by(node: ast.AST) -> set[str]:
    """Names a single statement binds in the scope that contains it."""
    out: set[str] = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        out.add(node.name)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for a in node.names:
            out.add(a.asname or a.name.split(".")[0])
    elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign,
                           ast.For, ast.AsyncFor, ast.With, ast.AsyncWith,
                           ast.Try, ast.While, ast.If, ast.Match)):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                out.add(sub.id)
            elif isinstance(sub, ast.alias):
                out.add(sub.asname or sub.name.split(".")[0])
            elif isinstance(sub, (ast.ExceptHandler,)) and sub.name:
                out.add(sub.name)
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                out.add(sub.name)
    return out


def _loads_before_binding(body: list[ast.stmt], known: set[str],
                          problems: list[str], where: str) -> None:
    """Walk statements in order; report Name loads nothing has bound yet."""
    seen = set(known)
    for stmt in body:
        # A function body runs later, so only its *signature* is evaluated
        # here; its interior is checked separately with the names that will
        # exist by call time. A class body, by contrast, runs immediately --
        # which is exactly the TrainerCallback failure -- so it is walked
        # inline against the names bound so far.
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in stmt.decorator_list:
                _check_expr(d, seen, problems, where)
            for d in (stmt.args.defaults + [x for x in stmt.args.kw_defaults
                                            if x is not None]):
                _check_expr(d, seen, problems, where)
            seen.add(stmt.name)
            continue
        if isinstance(stmt, ast.ClassDef):
            for b in stmt.bases + stmt.decorator_list:
                _check_expr(b, seen, problems, f"{where}class {stmt.name}")
            seen.add(stmt.name)
            inner = set(seen)
            _loads_before_binding(stmt.body, inner, problems,
                                  f"{where}{stmt.name}.")
            continue
        _check_expr(stmt, seen, problems, where)
        seen |= _bound_by(stmt)


def _check_expr(node: ast.AST, seen: set[str], problems: list[str],
                where: str) -> None:
    """Report Name loads in `node` that `seen` does not cover.

    Comprehensions and lambdas bind their own targets, and a name stored
    anywhere inside this statement (`x = x_source`, `for q in ...`) is fair
    game later in it, so both are added before looking.
    """
    local = set(seen)
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store,)):
            local.add(sub.id)
        elif isinstance(sub, ast.comprehension):
            for t in ast.walk(sub.target):
                if isinstance(t, ast.Name):
                    local.add(t.id)
        elif isinstance(sub, ast.Lambda):
            local |= {a.arg for a in sub.args.args + sub.args.kwonlyargs}
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)):
            local.add(sub.name)
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            local.add(sub.name)
        elif isinstance(sub, ast.alias):
            local.add(sub.asname or sub.name.split(".")[0])
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            if sub.id not in local and sub.id not in BUILTINS:
                problems.append(
                    f"line {sub.lineno}: {where}{sub.id!r} is read before "
                    f"anything binds it")


def _all_bound_in(body: list[ast.stmt]) -> set[str]:
    out: set[str] = set()
    for stmt in body:
        out |= _bound_by(stmt)
    return out


def _direct_functions(body: list[ast.stmt]):
    """Functions belonging to *this* scope, not every function beneath it.

    A plain `ast.walk` for FunctionDef reaches arbitrarily deep, so a method
    three scopes down gets checked against the module's names and every
    closure reads as an error. Class bodies are included because a method's
    enclosing scope, for this purpose, is the scope holding the class; the
    control-flow statements are included because `if`/`try`/`for` do not
    open a scope.
    """
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield stmt
        elif isinstance(stmt, ast.ClassDef):
            yield from _direct_functions(stmt.body)
        elif isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While,
                               ast.With, ast.AsyncWith, ast.Try)):
            yield from _direct_functions(stmt.body)
            yield from _direct_functions(getattr(stmt, "orelse", []))
            yield from _direct_functions(getattr(stmt, "finalbody", []))
            for h in getattr(stmt, "handlers", []):
                yield from _direct_functions(h.body)


def _check_function(fn: ast.FunctionDef | ast.AsyncFunctionDef,
                    outer: set[str], problems: list[str]) -> None:
    """Check one function body, then recurse into the functions inside it.

    By the time a function is *called*, its enclosing scopes have finished
    running -- so a nested definition may legitimately read anything the
    enclosing body binds, at any line. Checking nested functions against the
    module scope alone reported four such reads as errors on the first run
    of this file, all of them fine. Order still matters *within* each body,
    which is where the real failures have been.
    """
    args = fn.args
    entry = set(outer)
    entry |= {a.arg for a in args.args + args.kwonlyargs + args.posonlyargs}
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            entry.add(extra.arg)

    _loads_before_binding(fn.body, entry, problems, f"{fn.name}(): ")

    visible = entry | _all_bound_in(fn.body)
    for sub in _direct_functions(fn.body):
        _check_function(sub, visible, problems)


def check(path: Path) -> int:
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        print(f"{path}:{exc.lineno}: SyntaxError: {exc.msg}")
        return 1

    problems: list[str] = []
    _loads_before_binding(tree.body, set(), problems, "")

    module_names = _all_bound_in(tree.body)
    for sub in _direct_functions(tree.body):
        _check_function(sub, module_names, problems)

    seen, unique = set(), []
    for q in problems:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    for q in unique:
        print(f"{path}: {q}")
    print(f"{path}: {len(unique)} problem(s), {len(src.splitlines())} lines")
    return 1 if unique else 0


if __name__ == "__main__":
    targets = [Path(a) for a in sys.argv[1:]] or [Path("kaggle/01_sft.py")]
    sys.exit(max(check(t) for t in targets))
