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
