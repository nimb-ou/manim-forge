"""Assembly is the failure mode the split introduces, so it is what to test.

A single model never assembles anything, so none of these can go wrong for
it. Two beats naming the same variable, or a beat using one that no earlier
beat created, are new — and they would surface as confusing render errors
rather than as what they are.
"""
import pytest

from forge.app.twostage import (Beat, assemble, extract_code, parse_plan)

pytest.importorskip("manim")


def test_a_plan_window_parses():
    beats, ended = parse_plan(
        "1. [12s] A circle appears -- Here is a circle.\n"
        "2. [8s] It turns red -- Watch closely.")
    assert [b.n for b in beats] == [1, 2]
    assert beats[0].seconds == 12.0
    assert beats[1].intent == "It turns red"
    assert beats[1].narration == "Watch closely."
    assert not ended


def test_end_is_recognised():
    _, ended = parse_plan("1. [4s] Done -- That is all.\nEND")
    assert ended


def test_a_beat_without_narration_still_parses():
    beats, _ = parse_plan("1. [5s] A square appears")
    assert beats and beats[0].intent == "A square appears"


def test_prose_around_the_plan_is_ignored():
    beats, _ = parse_plan("Sure! Here is the plan:\n\n1. [3s] Open -- Hi.\n\n"
                          "Let me know if you want changes.")
    assert len(beats) == 1


def test_a_runaway_plan_is_capped():
    text = "\n".join(f"{i}. [5s] beat {i} -- x" for i in range(1, 200))
    beats, ended = parse_plan(text, limit=40)
    assert len(beats) == 40 and ended


def test_assembly_produces_a_parsing_scene():
    beats = [Beat(1, 5, "circle"), Beat(2, 5, "colour")]
    a = assemble(beats, ["c = Circle()\nself.play(Create(c))",
                         "self.play(c.animate.set_color(RED))"])
    assert a.ok, a.problems
    assert "class ForgeScene(Scene):" in a.code
    assert "# beat 1: circle" in a.code
    import ast
    ast.parse(a.code)


def test_a_beat_using_a_name_no_beat_defines_is_caught():
    """The failure only a multi-call pipeline can have."""
    a = assemble([Beat(1, 5, "one"), Beat(2, 5, "two")],
                 ["c = Circle()", "self.play(sq.animate.shift(UP))"])
    assert not a.ok
    assert "sq" in a.problems[0]


def test_manim_names_are_not_reported_missing():
    a = assemble([Beat(1, 5, "x")],
                 ["self.play(Create(Circle(color=BLUE)), run_time=2)"])
    assert a.ok, a.problems


def test_comprehension_and_loop_targets_are_not_missing():
    a = assemble([Beat(1, 5, "x")],
                 ["dots = [Dot() for i in range(3)]\n"
                  "for d in dots:\n    self.add(d)"])
    assert a.ok, a.problems


def test_unparseable_beats_are_reported_not_rendered():
    a = assemble([Beat(1, 5, "x")], ["this is ( not python"])
    assert not a.ok and "does not parse" in a.problems[0]


def test_empty_beats_are_dropped_not_assembled_blank():
    a = assemble([Beat(1, 5, "x"), Beat(2, 5, "y")],
                 ["c = Circle()\nself.add(c)", "   "])
    assert a.ok, a.problems


def test_extract_code_takes_the_fenced_block():
    assert extract_code("blah\n```python\nx = 1\n```\nmore") == "x = 1"
    assert extract_code("x = 1") == "x = 1"


def test_a_plan_without_brackets_still_parses():
    """What the untuned model actually writes.

    It omits the bracketed duration and copies the word "intent" out of the
    format spec. Scoring that as zero beats reads as "the planner produced
    nothing", which is the opposite of what happened.
    """
    beats, ended = parse_plan(
        "1. 0.000 intent -- Start with a blank screen.\n"
        "2. 0.500 intent -- Display a single dot.\nEND")
    assert len(beats) == 2 and ended
    assert beats[0].seconds == 0.0
    assert beats[1].narration == "Display a single dot."


def test_no_beats_is_reported_as_itself():
    a = assemble([], [])
    assert not a.ok and a.problems == ["no beats to assemble"]
