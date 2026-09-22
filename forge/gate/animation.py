"""Measure whether a scene animates and explains, or merely displays.

The render gate answers *did this produce a video file*. Nothing in this
project has answered *is this an explanation*, because a slideshow renders
perfectly well.

So this measures structure, never taste, and never with a model judge: the
44 hand-written gold scenes are a labelled positive set, which makes a judge
unnecessary and would anyway be a second thing to debug that moves whenever
it moves.

**What the first version of this file assumed, and what measuring said.**
Scored against 44 gold scenes and 2,770 non-gold corpus rows, per-signal AUC
(0.5 is a coin flip):

    asserts            0.955      40/44 gold, 0/2770 corpus
    computed_labels    0.926
    text_objs          0.879      gold has *more* text, not less
    n_play             0.844
    declared_seconds   0.813
    discrete/100calls  0.226      gold animates less per unit of work
    continuous         0.538      <- coin flip
    continuous/play    0.535      <- coin flip
    transforms         0.462      <- coin flip
    geom_objs          0.534      <- coin flip

Three of those overturned assumptions the design rested on.

*Continuous constructs do not separate a gold scene from a corpus row.* The
recorded finding -- ValueTracker in 9.1% of gold against 1.4% of the corpus
-- is true, and is a corpus-level ratio resting on **four scenes**. The
other 40 gold scenes do not use ValueTracker either. As a per-scene
discriminator it is noise, and this gate was designed around it.

*Text-heavy does not mean slideware.* Gold averages 16.4 text mobjects
against the corpus's 7.7. 3Blue1Brown scenes label things. The geometry
component was removed rather than inverted to fit, because inverting a
signal to match the labels is fitting, not measuring.

*`assert` separates perfectly because it is a house convention.* 40 of 44
gold scenes assert; no corpus row does. `assert True` would score full
marks. It is a target worth teaching -- it forces a scene to compute what it
claims -- but it is not evidence about animation, so it is reported as
`convention` and kept out of `score`.

**The components that survived.**

    pacing      discrete animations per 100 calls. Gold 13.4, corpus 19.7:
                gold scenes put more construction behind each animation
    grounding   f-strings and .set_value inside text mobjects -- a number
                the scene worked out rather than one someone typed
    structure   beats, and whether play calls are spread through the code
                or bunched at the end
    duration    declared run_time and waits; slideware is short because
                nothing takes time to happen
    substance   play calls

Composite AUC is 0.98 against the corpus, stable across disjoint samples
(0.970-0.979), and both halves of the gold set separate equally (0.978 /
0.967) -- so the component choice is not fitted to one subset. About a
quarter of corpus rows still reach the weakest gold scene, which is the
honest ceiling of this version.

**No component may divide by anything filler can move.** Twice a test caught
five `assert True`s raising the score: first through `loc` in pacing, then
through `loc` in the play-call spread. Phase 3 turns this measure into a
GRPO reward, and a reward you can raise with filler is a reward for filler.

Usage:
    from forge.gate.animation import profile, score
    p = profile(source_code)
    p.score          # 0..1 composite
    p.convention     # the assert habit, reported and never scored
    p.as_dict()      # every raw count behind both
"""
from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field

# ── vocabulary ─────────────────────────────────────────────────────────────
# Continuous: the value changes over time and the screen follows it.
CONTINUOUS_NAMES = {
    "ValueTracker", "DecimalNumber", "always_redraw", "add_updater",
    "UpdateFromFunc", "UpdateFromAlphaFunc", "MoveAlongPath", "Rotating",
    "ChangingDecimal", "ComplexValueTracker", "TracedPath", "Succession",
    "ApplyMethod", "MoveToTarget", "Homotopy", "ApplyPointwiseFunction",
}
# Transforms relate a before to an after: still motion, weaker than a tracker.
TRANSFORM_NAMES = {
    "Transform", "ReplacementTransform", "TransformMatchingTex",
    "TransformMatchingShapes", "FadeTransform", "ClockwiseTransform",
    "CounterclockwiseTransform", "CyclicReplace", "Swap", "Restore",
}
# Discrete: something appears or vanishes. A slideshow is made of these.
DISCRETE_NAMES = {
    "FadeIn", "FadeOut", "Write", "Unwrite", "Create", "Uncreate",
    "DrawBorderThenFill", "ShowIncreasingSubsets", "AddTextLetterByLetter",
    "SpiralIn", "GrowFromCenter", "GrowFromEdge", "GrowFromPoint",
    "ShrinkToCenter", "FadeInFromPoint", "RemoveTextLetterByLetter",
}
TEXT_NAMES = {
    "Text", "Tex", "MathTex", "Title", "MarkupText", "Paragraph",
    "BulletedList", "Code", "DecimalNumber", "Integer", "Variable",
}
GEOMETRY_NAMES = {
    "Circle", "Square", "Rectangle", "Triangle", "Polygon", "RegularPolygon",
    "Line", "Arrow", "Vector", "DoubleArrow", "Dot", "Arc", "ArcBetweenPoints",
    "Ellipse", "Annulus", "Sector", "Axes", "NumberPlane", "NumberLine",
    "ThreeDAxes", "Surface", "Sphere", "Cube", "Cylinder", "Cone", "Torus",
    "ParametricFunction", "FunctionGraph", "ImplicitFunction", "Brace",
    "Angle", "RightAngle", "CurvedArrow", "Polyhedron", "Dodecahedron",
    "VectorField", "StreamLines", "Graph", "BarChart",
}


@dataclass
class Profile:
    """Every count behind the score, so the score can be argued with."""

    parsed: bool = True
    n_play: int = 0
    n_wait: int = 0
    n_sections: int = 0
    continuous: int = 0
    transforms: int = 0
    discrete: int = 0
    animate_calls: int = 0
    text_objs: int = 0
    geom_objs: int = 0
    asserts: int = 0
    computed_labels: int = 0
    declared_seconds: float = 0.0
    n_calls: int = 0
    play_line_spread: float = 0.0
    loc: int = 0
    scores: dict = field(default_factory=dict)
    score: float = 0.0
    convention: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def _name_of(node: ast.AST) -> str | None:
    """The callable's bare name, through attribute access."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def profile(source: str) -> Profile:
    """Count the structural habits of one scene. Never executes the code."""
    p = Profile()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        p.parsed = False
        return p

    p.loc = len([l for l in source.splitlines() if l.strip()
                 and not l.strip().startswith("#")])
    play_lines: list[int] = []
    call_lines: list[int] = []
    # Identity of each `.animate` attribute seen. `x.animate.scale(2).set_color(RED)`
    # is two Call nodes whose chains both reach the *same* animate, and
    # counting per call made one use look like two -- which would inflate the
    # motion score exactly on the scenes that use animate most fluently.
    animate_nodes: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            p.asserts += 1
        if not isinstance(node, ast.Call):
            continue
        p.n_calls += 1
        call_lines.append(getattr(node, 'lineno', 0))
        name = _name_of(node)
        if name is None:
            continue

        if name == "play":
            p.n_play += 1
            play_lines.append(getattr(node, "lineno", 0))
            for kw in node.keywords:
                if kw.arg == "run_time" and isinstance(kw.value, ast.Constant) \
                        and isinstance(kw.value.value, (int, float)):
                    p.declared_seconds += float(kw.value.value)
        elif name == "wait":
            p.n_wait += 1
            if node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, (int, float)):
                p.declared_seconds += float(node.args[0].value)
            elif not node.args:
                p.declared_seconds += 1.0
        elif name in ("next_section", "add_section", "beat"):
            # `beat` because the gold scenes never call next_section directly:
            # they carry @beat(...) decorators and ForgeScene issues the
            # section call for them. Counting only the literal call read 0.00
            # sections for all 44 gold scenes and 0.00 for the corpus, so the
            # structure signal was comparing nothing to nothing while looking
            # like a measurement.
            p.n_sections += 1

        if name in CONTINUOUS_NAMES:
            p.continuous += 1
        elif name in TRANSFORM_NAMES:
            p.transforms += 1
        elif name in DISCRETE_NAMES:
            p.discrete += 1
        if name in TEXT_NAMES:
            p.text_objs += 1
        elif name in GEOMETRY_NAMES:
            p.geom_objs += 1

        # `.animate` is an attribute chain, not a call name -- the node for
        # `sq.animate.shift(RIGHT)` has func=Attribute(attr='shift'), and
        # 'animate' sits one level down. So walk down the chain looking for
        # it.
        #
        # The first version of this loop reassigned the cursor to itself
        # whenever the chain ended in a plain Name -- `self.play(...)`, which
        # is every scene -- and hung. Each step here moves strictly downward
        # or stops, which is the property that was missing.
        if isinstance(node.func, ast.Attribute):
            cur: ast.AST = node.func
            while isinstance(cur, ast.Attribute):
                if cur.attr == "animate":
                    animate_nodes.add(id(cur))
                    break
                nxt = cur.value
                if isinstance(nxt, ast.Call):
                    nxt = nxt.func
                cur = nxt

    p.animate_calls = len(animate_nodes)

    # A label whose text is built from a value rather than typed: f-strings
    # and %/format inside a text mobject. `MathTex(f"{area:.2f}")` is a number
    # the scene worked out; `MathTex("3.14")` is a number someone typed.
    p.computed_labels = len(re.findall(
        r"(?:Text|Tex|MathTex|DecimalNumber|MarkupText|Title)\s*\(\s*[rf]{1,2}[\"']",
        source))
    p.computed_labels += len(re.findall(r"\.set_value\s*\(", source))

    if len(play_lines) >= 2 and len(call_lines) >= 2:
        # How much of the *code* the play calls are spread over. A scene that
        # builds, shows, and moves on has them throughout; slideware bunches
        # them at the end after a long construction block.
        #
        # Measured against the span of the code rather than the line count,
        # because line count is paddable: five trailing `assert True`s used
        # to lower this without changing where a single play call sat. A
        # test caught it, which is the second time the same filler found a
        # different way in -- so the rule is that no component may divide by
        # anything a blank-ish line can move.
        span = max(call_lines) - min(call_lines)
        p.play_line_spread = _clip(
            (max(play_lines) - min(play_lines)) / max(span, 1))

    # ── what the first version got wrong ──────────────────────────────
    # Measured against the 44 gold scenes and 2,770 non-gold corpus rows,
    # per-signal AUC (0.5 is a coin flip):
    #
    #   asserts            0.955   40/44 gold, 0/2770 corpus
    #   computed_labels    0.926
    #   text_objs          0.879   gold has *more* text, not less
    #   n_play             0.844
    #   declared_seconds   0.813
    #   discrete/100loc    0.204   gold animates *less* per line
    #   continuous         0.538   <- coin flip
    #   continuous/play    0.535   <- coin flip
    #   transforms         0.462   <- coin flip
    #   geom_objs          0.534   <- coin flip
    #
    # Two of those overturn assumptions this project was built on.
    #
    # **Continuous constructs do not separate a gold scene from a corpus
    # row.** The recorded finding -- ValueTracker in 9.1% of gold against
    # 1.4% of the corpus -- is true and is a *corpus-level ratio* over four
    # scenes. 40 of the 44 gold scenes do not use ValueTracker either, so as
    # a per-scene discriminator it is noise, and Phase 2 was designed around
    # it.
    #
    # **Text objects point the other way.** Gold averages 16.4 text mobjects
    # against the corpus's 7.7. 3Blue1Brown scenes label things. "Text-heavy
    # means slideware" does not survive contact with the positive set, so
    # the geometry component is gone rather than inverted to fit.
    #
    # And `asserts` separates perfectly *because it is a house convention*,
    # not because asserted scenes animate better. `assert True` would score
    # full marks. It is a target worth teaching -- it forces a scene to
    # compute what it claims -- but it is not evidence about animation, so
    # it is reported on its own and kept out of the animation score.
    # Denominated in *calls*, not lines. Lines are paddable: five
    # `assert True`s raised the old pacing score without changing a single
    # thing about the animation, which a test caught and which matters more
    # than it looks -- Phase 3 turns this measure into a GRPO reward, and a
    # reward you can raise with filler is a reward for filler. Padding the
    # call count means writing code that constructs something.
    static_rate = p.discrete / max(p.n_calls, 1) * 100
    p.scores = {
        # Not a continuous-vs-discrete ratio: gold scenes have *more*
        # discrete calls in absolute terms because they are 2.4x longer.
        # What separates them is how much construction sits behind each
        # animation event. Gold averages 13.4 discrete animations per 100
        # calls against the corpus's 19.7, so 30 is roughly the point at
        # which a scene is doing nothing but appearing and disappearing
        # things. The divisor moved from 12 to 30 when the denominator
        # changed from lines to calls; left at 12 it clipped every gold
        # scene to zero, and the composite hid that because four other
        # components carried it.
        "pacing": _clip(1.0 - static_rate / 30.0),
        # f-strings and .set_value inside text mobjects: a number the scene
        # worked out rather than one someone typed.
        "grounding": _clip(p.computed_labels / 6.0),
        "structure": _clip(0.6 * _clip(p.n_sections / 4.0)
                           + 0.4 * p.play_line_spread),
        "duration": _clip(p.declared_seconds / 45.0),
        "substance": _clip(p.n_play / 18.0),
    }
    weights = {"pacing": 0.25, "grounding": 0.25, "structure": 0.20,
               "duration": 0.15, "substance": 0.15}
    p.score = round(sum(p.scores[k] * w for k, w in weights.items()), 4)
    # Reported, never scored. It says "written to our convention", which is
    # worth knowing and is not the same question.
    p.convention = round(_clip(p.asserts / 3.0), 4)
    p.scores = {k: round(v, 4) for k, v in p.scores.items()}
    return p


def score(source: str) -> float:
    return profile(source).score
