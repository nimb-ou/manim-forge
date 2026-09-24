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
    """Capped, but not reported as finished.

    This test used to assert `ended` here, which encoded the bug it was
    supposed to guard: the driver reads `ended` as "the planner is done" and
    stops asking for more, so a cap that claims to be END truncates every
    arc at one window.
    """
    text = "\n".join(f"{i}. [5s] beat {i} -- x" for i in range(1, 200))
    beats, ended = parse_plan(text, limit=40)
    assert len(beats) == 40
    assert not ended


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


def test_the_chat_end_token_is_stripped():
    """What actually broke the untuned coder baseline.

    mlx-lm returns the template's end-of-turn token inside the generated
    string, so `circle.become(square)<|im_end|>` was reported as the coder
    writing invalid syntax. The statement was correct; the harness was
    breaking it.
    """
    assert extract_code("circle.become(square)<|im_end|>") == "circle.become(square)"
    assert extract_code("```python\nx = 1\n```<|im_end|>") == "x = 1"
    for tokname in ("<|endoftext|>", "<|eot_id|>", "<|end_of_text|>"):
        assert extract_code(f"y = 2{tokname}") == "y = 2"


def test_text_after_the_end_token_is_dropped():
    assert extract_code("a = 1<|im_end|>\nassistant\nb = 2") == "a = 1"


def test_a_scene_that_never_waits_gets_a_hold():
    """Otherwise it renders zero frames and reports `empty_render`.

    A single model writes its own ending; a concatenation of beats has none
    unless one is added.
    """
    a = assemble([Beat(1, 5, "x")], ["c = Circle()\nself.add(c)"])
    assert a.ok and a.code.rstrip().endswith("self.wait(1)")


def test_a_scene_that_already_waits_is_left_alone():
    a = assemble([Beat(1, 5, "x")],
                 ["c = Circle()\nself.play(Create(c))\nself.wait(2)"])
    assert a.code.count("self.wait") == 1


def test_hitting_the_limit_is_not_the_planner_saying_END():
    """The bug that capped every hard-eval plan at six beats.

    The driver asks for a window, the parser caps it, and if the cap reports
    itself as END the driver stops asking. Real arcs run about 40 beats.
    """
    text = "\n".join(f"{i}. [20s] beat {i} -- narration" for i in range(1, 12))
    beats, ended = parse_plan(text, limit=6)
    assert len(beats) == 6
    assert not ended, "the cap must not look like the planner finishing"
    beats, ended = parse_plan(text + "\nEND", limit=20)
    assert ended


def test_self_attributes_count_as_names_in_scope():
    """The gold scenes carry state between beats on `self`.

    Tracking only plain locals told the coder the scene had no names in it,
    while it was writing `self.axes` — so it invented attributes and every
    assembled scene died on api_misuse.
    """
    from forge.app.twostage import names_in_scope
    scope = names_in_scope(["self.axes = Axes()\nc = Circle()\nself.add(c)"])
    assert "self.axes" in scope and "c" in scope


def test_an_unset_self_attribute_is_caught():
    a = assemble([Beat(1, 5, "x"), Beat(2, 5, "y")],
                 ["self.axes = Axes()\nself.play(Create(self.axes))",
                  "self.play(FadeOut(self.grid))"])
    assert not a.ok and "self.grid" in a.problems[0]


def test_scene_own_attributes_are_not_reported_missing():
    """Manim's Scene supplies camera, renderer, mobjects and the rest."""
    a = assemble([Beat(1, 5, "x")],
                 ["self.play(Create(Circle()))\nself.add(Square())\n"
                  "self.wait()"])
    assert a.ok, a.problems


def test_failing_beat_maps_rich_marker_to_beat():
    from forge.app.twostage import failing_beat
    code = ("class S(Scene):\n    def construct(self):\n"
            "        # beat 1: a\n        c = Circle()\n"
            "        # beat 2: b\n        self.camera.frame.scale(0.8)\n")
    err = ("│ ❱ 6 │   │   self.camera.frame.scale(0.8)   │\n"
           "│ ❱ 187 │   raise TypeError(\"x\")   │\n")
    assert failing_beat(code, err) == 2
    assert failing_beat(code, "no markers") is None


def test_beat_prompt_shape():
    from forge.app.twostage import Beat, beat_prompt
    beats = [Beat(1, 20.0, "a circle", "say hi"), Beat(2, None, "it grows")]
    p = beat_prompt("circles", beats, 1, ["c = Circle()\nself.play(Create(c))"])
    assert "ALREADY ON SCREEN\n  1. a circle" in p
    assert "NAMES IN SCOPE\n  c" in p
    assert "HELPERS THIS SCENE DEFINES\n  (none)" in p
    assert p.endswith("WRITE THIS BEAT — step 2 of 2\n  intent: it grows")


def test_missing_names_and_prelude_prompt():
    from forge.app.twostage import Beat, missing_names, prelude_prompt
    beats = [Beat(1, None, "a graph"), Beat(2, None, "a point")]
    bodies = ["g = axes.plot(lambda x: x**2)\nself.play(Create(g))",
              "d = Dot(axes.c2p(1, 1))\nself.play(FadeIn(d))"]
    assert missing_names(beats, bodies) == ["axes"]
    p = prelude_prompt("parabola", bodies, ["axes"])
    assert "axes.plot(lambda x: x**2)" in p and "Do not call" in p


def test_intent_key_catches_counting_loops():
    from forge.app.twostage import intent_key
    assert intent_key("Span of three vectors.") == intent_key("Span of 11 vectors")
    assert intent_key("A circle") != intent_key("A square")


def test_intent_key_keeps_ordinals():
    from forge.app.twostage import intent_key
    assert intent_key("The first derivative") != intent_key("The second derivative")


def test_parsing_prefix_keeps_lines_before_a_truncation():
    from forge.app.twostage import parsing_prefix
    code = "c = Circle()\nself.play(Create(c))\nself.play(c.animate.shift(RIGHT"
    assert parsing_prefix(code) == "c = Circle()\nself.play(Create(c))"
    assert parsing_prefix("(((") == ""
