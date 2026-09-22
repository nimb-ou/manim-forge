"""The animation measure, held to the things that went wrong building it.

Every test here is a mistake the first version made. The measure is the
foundation of Phase 2 — the mix gets rebuilt against it — so a measure that
quietly stops discriminating would not announce itself.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def m():
    spec = importlib.util.spec_from_file_location(
        "forge_animation", ROOT / "forge" / "gate" / "animation.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["forge_animation"] = mod
    spec.loader.exec_module(mod)
    return mod


SLIDEWARE = """
from manim import *
class S(Scene):
    def construct(self):
        a = Text("One"); self.play(Write(a)); self.play(FadeOut(a))
        b = Text("Two"); self.play(Write(b)); self.play(FadeOut(b))
        c = Text("Three"); self.play(Write(c)); self.play(FadeOut(c))
"""


def test_the_animate_walk_terminates(m):
    """The first version hung on `self.play(...)`.

    Its cursor reassigned to itself whenever an attribute chain ended in a
    plain Name, which is every scene ever written. It looked like a slow
    import for two runs.
    """
    src = "from manim import *\n" \
          "class S(Scene):\n" \
          "    def construct(self):\n" \
          "        self.play(VGroup(Square()).animate.scale(2).set_color(RED))\n"
    p = m.profile(src)
    assert p.parsed and p.n_play == 1


def test_a_chained_animate_counts_once(m):
    """`x.animate.scale(2).set_color(RED)` is one use, not two.

    Counting per Call node inflated the signal exactly on the scenes using
    animate most fluently.
    """
    src = ("from manim import *\nclass S(Scene):\n    def construct(self):\n"
           "        self.play(Square().animate.scale(2).set_color(RED).shift(UP))\n")
    assert m.profile(src).animate_calls == 1


def test_asserts_do_not_move_the_score(m):
    """40/44 gold scenes assert and 0/2770 corpus rows do.

    That is a house convention, so it separates perfectly and says nothing
    about animation — `assert True` would score full marks. It is reported
    as `convention` and must stay out of `score`, or the gate rewards a
    habit rather than a quality.
    """
    plain = m.profile(SLIDEWARE)
    asserted = m.profile(SLIDEWARE + "        assert True\n" * 5)
    assert asserted.score == plain.score
    assert asserted.convention > plain.convention


def test_slideware_scores_low(m):
    assert m.profile(SLIDEWARE).score < 0.30


def test_unparseable_code_is_not_silently_zero_scored(m):
    p = m.profile("this is (not python")
    assert p.parsed is False and p.score == 0.0


def test_gold_still_separates_from_the_corpus(m):
    """The whole point, as an integration test.

    Phase 2's exit condition is a visible gap. If this slips below 0.9 the
    measure has stopped discriminating and the mix must not be rebuilt
    against it.
    """
    spec = importlib.util.spec_from_file_location(
        "gap", ROOT / "scripts" / "animation_gap.py")
    gap = importlib.util.module_from_spec(spec)
    sys.modules["gap"] = gap
    spec.loader.exec_module(gap)
    if not (ROOT / "data" / "gold" / "gold.jsonl").exists():
        pytest.skip("gold set not present")
    g = [m.profile(c).score for _, c in gap.gold_rows()]
    c = [m.profile(code).score for _, code in gap.corpus_rows(200, seed=5)]
    assert gap.auc(g, c) > 0.90, f"AUC fell to {gap.auc(g, c):.3f}"


def test_the_corpus_sample_excludes_gold(m):
    """240 of the mix's 3010 rows are gold; sampling them as negatives
    scores part of the positive set as the thing it is meant to beat."""
    spec = importlib.util.spec_from_file_location(
        "gap", ROOT / "scripts" / "animation_gap.py")
    gap = importlib.util.module_from_spec(spec)
    sys.modules["gap"] = gap
    spec.loader.exec_module(gap)
    ids = [i for i, _ in gap.corpus_rows(500, seed=3)]
    assert not [i for i in ids if i.startswith("gold")]
