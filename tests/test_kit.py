"""The kit's contract with the model: every block is documented where the
model reads (KIT_API), named as a verb, and takes the stage first."""
import inspect
import re

from forge.kit import kit


def test_every_block_is_in_the_api_the_model_sees():
    listed = set(re.findall(r"\b([a-z_]+)\(stage", kit.KIT_API))
    missing = kit.KIT_BLOCKS - listed
    assert not missing, f"blocks the model is never told about: {sorted(missing)}"


def test_blocks_take_the_stage_first_and_have_docstrings():
    for name in kit.KIT_BLOCKS:
        f = getattr(kit, name)
        assert list(inspect.signature(f).parameters)[0] == "stage", name
        assert (f.__doc__ or "").strip(), f"{name} has no docstring"


def test_moves_are_blocks():
    assert kit.KIT_MOVES <= kit.KIT_BLOCKS | {"swap_bars"}


def test_no_block_is_named_like_a_noun_a_model_would_assign():
    # A model writes `plane = ...`, `axes = ...`, `graph = ...`; a block of
    # that name is then shadowed. Blocks are verbs.
    nouns = {"plane", "vector", "axes", "graph", "area", "wave", "bars",
             "network", "grid", "circle", "square"}
    assert not (kit.KIT_BLOCKS & nouns)


def test_kit_embeds_into_an_assembled_scene():
    from forge.app.twostage import Beat, assemble
    asm = assemble([Beat(1, None, "a grid")],
                   ["p = draw_plane(stage)\nv = draw_vector(stage, p, (1, 2))"],
                   kit=True)
    assert asm.ok, asm.problems
    assert "stage = Stage(self)" in asm.code
    bad = assemble([Beat(1, None, "x")], ["draw_vector(stage, q, (1, 2))"], kit=True)
    assert any("q" in p for p in bad.problems)


def test_spaced_maths_labels_are_latex_and_words_stay_text(monkeypatch):
    # Which constructor _text picks (CI has no LaTeX, so a real MathTex
    # falls back to Text there).
    monkeypatch.setattr(kit, "MathTex", lambda s, **k: ("MathTex", s))
    monkeypatch.setattr(kit, "Tex", lambda s, **k: ("Tex", s))
    monkeypatch.setattr(kit, "Text", lambda s, **k: ("Text", s))
    st = kit.Stage.__new__(kit.Stage)
    for s in (r"\pi r", r"2\pi r", r"\theta = 30"):
        assert st._text(s, 28)[0] == "MathTex", s
    for s in ("day 0", r"the \pi slice", "Legs 3 and 4"):
        assert st._text(s, 28)[0] == "Text", s


def test_axis_and_curve_labels_keep_words_and_maths_apart(monkeypatch):
    monkeypatch.setattr(kit, "MathTex", lambda s, **k: ("MathTex", s))
    monkeypatch.setattr(kit, "Text", lambda s, **k: ("Text", s))
    for s in ("x", "x^2", r"\cos x", "guess"):
        assert kit._axis_tag(s)[0] == "MathTex", s
    for s in ("more demand", "£k income", "m/s²"):
        assert kit._axis_tag(s)[0] == "Text", s


def test_middle_pictures_step_aside_for_an_equation_on_screen(monkeypatch):
    class Scene:
        mobjects = []
    st = kit.Stage.__new__(kit.Stage)
    line = object()
    st.scene, st._equation, st._equation_where = Scene(), [line], "right"
    monkeypatch.setattr(kit, "_CURRENT", [st])
    assert kit._REGIONS["center"] == dict.__getitem__(kit._REGIONS, "center")  # none on screen
    Scene.mobjects = [line]
    assert kit._REGIONS["center"] == dict.__getitem__(kit._REGIONS, "left")
    assert kit._REGIONS["right"] == dict.__getitem__(kit._REGIONS, "right")
    st._equation_where = "left"
    assert kit._REGIONS["full"] == dict.__getitem__(kit._REGIONS, "right")


def test_draw_vector_takes_a_from_to_pair(monkeypatch):
    # draw_vector(stage, p, (0, 0), (3, 4)) once drew an arrow to (0, 0).
    made = []
    monkeypatch.setattr(kit, "Arrow", lambda a, b, **k: made.append((tuple(a), tuple(b), k)) or object())
    monkeypatch.setattr(kit, "GrowArrow", lambda a, **k: None)

    class Plane:
        def c2p(self, x, y):
            return (float(x), float(y), 0.0)

    class Scene:
        def play(self, *a, **k):
            pass
    st = kit.Stage.__new__(kit.Stage)
    st.scene, st._objects, st._issues = Scene(), [], 0
    kit.draw_vector.__wrapped__(st, Plane(), (1, 1), (3, 2))
    assert made[0][0][:2] == (1.0, 1.0) and made[0][1][:2] == (3.0, 2.0)
    assert made[0][2]["color"] == kit.YELLOW


def test_kit_issues_are_counted_and_said(capsys):
    st = kit.Stage.__new__(kit.Stage)
    st._issues, st._t = 0, [1.0, 2.0]
    st.issue("the point (9, 9) is outside its axes")
    assert st._issues == 1
    assert "ISSUE 3 the point (9, 9) is outside its axes" in capsys.readouterr().err
