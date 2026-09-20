"""Lint rules: the repairs that need no model call.

Each rule is conservative by design -- it fires on one detectable condition
and its output is discarded unless it still parses. These tests pin the
conditions, and especially the cases a rule must *not* fire on.
"""
from __future__ import annotations

import pytest

from forge.repair.lint import (RULES, add_manim_import, has_bare_percent,
                               lint, needs_hold, needs_manim_import)

SCENE = "class A(Scene):\n    def construct(self):\n        self.play(Create(Dot()))\n"


def test_rule_names_are_unique_and_ordered():
    names = [n for n, _, _ in RULES]
    assert len(set(names)) == len(names)
    # strip_prose runs first: every other rule parses the code.
    assert names[0] == "strip_prose"


# --- add_manim_import ------------------------------------------------------

@pytest.mark.parametrize("code", [
    SCENE,
    '"""A module docstring."""\n' + SCENE,
    "from __future__ import annotations\n" + SCENE,
    "import math\n" + SCENE,
    "class B(ThreeDScene):\n    def construct(self):\n        self.wait()\n",
])
def test_fires_when_a_scene_has_no_manim_namespace(code):
    assert needs_manim_import(code)
    fixed, fired = lint(code)
    assert "add_manim_import" in fired
    assert "from manim import *" in fixed


@pytest.mark.parametrize("code", [
    "from manim import *\n" + SCENE,
    "from manim import Scene, Dot, Create\n" + SCENE,
    "import manim\n" + SCENE,
    pytest.param("from manimlib import *\n" + SCENE, id="manimgl-is-a-different-library"),
    pytest.param("class A(Scene:\n    oops\n", id="unparseable-fragment"),
    pytest.param("class A(object):\n    pass\n", id="not-a-scene"),
])
def test_does_not_fire(code):
    assert not needs_manim_import(code)


def test_docstring_stays_first():
    """Inserting above line 0 would demote the docstring to a bare string."""
    import ast
    out = add_manim_import('"""doc."""\n' + SCENE)
    assert ast.get_docstring(ast.parse(out)) == "doc."


def test_future_import_stays_first():
    """`from __future__` anywhere but the top is a SyntaxError."""
    import ast
    out = add_manim_import("from __future__ import annotations\n" + SCENE)
    ast.parse(out)                       # must not raise
    assert out.splitlines()[0].startswith("from __future__")


# --- the others ------------------------------------------------------------

def test_add_hold_only_when_nothing_moves():
    assert needs_hold("class A(Scene):\n    def construct(self):\n        self.add(Dot())\n")
    assert not needs_hold(SCENE)


def test_strip_prose_removes_a_leading_sentence():
    fixed, fired = lint("Here is the scene you asked for:\nfrom manim import *\n" + SCENE)
    assert "strip_prose" in fired
    assert fixed.startswith("from manim import *")


def test_escape_percent_leaves_non_tex_strings_alone():
    code = 'from manim import *\nx = "50% off"\nlabel = Tex("100\\\\%")\n'
    assert not has_bare_percent(code)


def test_escape_percent_fires_inside_a_tex_literal():
    code = 'from manim import *\nlabel = MathTex("50%")\n'
    assert has_bare_percent(code)
    fixed, fired = lint(code)
    assert "escape_percent" in fired
    assert "\\%" in fixed


def test_lint_never_returns_unparseable_code():
    """A rule whose output fails to parse is discarded."""
    import ast
    for code in [SCENE, "garbage {{{", "", "from manim import *\n"]:
        out, _ = lint(code)
        if code.strip() and "{{{" not in code:
            ast.parse(out)
