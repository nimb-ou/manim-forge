"""The API checker must stay precise, because its value is that it is.

It catches 1 of run 17's 23 failures. That is a small return, and it is only
worth anything at all because it has zero false positives across the 77
passes — a pre-render check that flags working scenes would cost more
attention than it saves. These tests pin both ends.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("manim")
from forge.repair.apicheck import briefing, check  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCENE = "from manim import *\nclass S(Scene):\n    def construct(self):\n        {}\n"


def test_an_unknown_name_is_caught():
    found = check(SCENE.format("x = Wobbulator()"))
    assert found and "Wobbulator" in found[0].message


def test_a_near_miss_gets_a_suggestion():
    found = check(SCENE.format("x = Dodecahedon()"))
    assert found and "Dodecahedron" in found[0].message


def test_builtins_are_not_manim_names():
    """`dir(__builtins__)` is dict methods inside a module, not builtins.

    That shipped for one revision and reported "Manim has no 'range'. Did
    you mean Triangle?" on four working scenes.
    """
    for call in ("x = range(3)", "x = enumerate([1])", "x = len([1])",
                 "x = zip([1],[2])", "x = sorted([1])"):
        assert check(SCENE.format(call)) == [], call


def test_locally_defined_names_shadow_manim():
    src = ("from manim import *\ndef Circle(**kw):\n    return None\n"
           "class S(Scene):\n    def construct(self):\n        Circle(nonsense=1)\n")
    assert check(src) == []


def test_a_correct_scene_is_silent():
    assert briefing(SCENE.format(
        "self.play(Create(Circle(radius=2, color=RED)))")) == ""


def test_unparseable_code_says_nothing():
    assert check("this is (not python") == []


def test_no_false_positives_on_the_recorded_passes():
    """The property that makes it usable at all."""
    p = ROOT / "data" / "bench" / "tuned100_n100_r4.json"
    if not p.exists():
        pytest.skip("no benchmark artefact")
    d = json.loads(p.read_text())
    trials = d["trials"] if isinstance(d, dict) else d
    flagged = [t["index"] for t in trials
               if t.get("ok") and check(t.get("code", ""))]
    assert flagged == [], f"flagged working scenes: {flagged}"
