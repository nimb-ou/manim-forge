"""Gold scene 003 — the dot product as a projection.

Driven by ``forge.primitives.vectors``. The projection drawn on screen is the
one ``projection_onto`` computes, the number shown is the one ``dot`` returns,
and the identity a·b = |a||b|cos(theta) is checked numerically before the scene
asserts it.

The last point is the one worth insisting on. A scene may render perfectly
while quietly showing a projection drawn by eye, and nobody watching could name
what is wrong — they would only feel that it was off. A model trained on
scenes whose geometry is decorative learns to produce decoration.
"""

import math

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.vectors import Vec2, dot_identity_check

A_C = BLUE_C
B_C = YELLOW_C
PROJ_C = "#5CD0B3"
NEG_C = "#FC6255"

ORIGIN_SHIFT = LEFT * 2.9 + DOWN * 0.6
PANEL_X = 3.5


class DotProduct(ForgeScene):

    # Chosen so the shadow falls to ~55% of |b|. The projection lies *along*
    # b by construction, so vectors where the shadow nearly reaches b's tip
    # draw two arrows on top of each other and the picture reads as one.
    a = Vec2(4.0, 0.5)
    b = Vec2(1.2, 3.2)

    def pt(self, v: Vec2):
        return ORIGIN_SHIFT + np.array([v.x, v.y, 0.0])

    def arrow(self, v: Vec2, color, width=5):
        return Arrow(ORIGIN_SHIFT, self.pt(v), buff=0, color=color,
                     stroke_width=width, max_tip_length_to_length_ratio=0.16)

    def panel(self, *mobs, y=1.5):
        return VGroup(*mobs).arrange(DOWN, buff=0.28,
                                     aligned_edge=LEFT).move_to([PANEL_X, y, 0])

    @beat("Draw two vectors from a shared origin", seconds=14,
          narration="Take two vectors sharing an origin. The dot product is "
                    "usually introduced as an arithmetic rule, but it has a "
                    "shape — and once you see the shape, the formula stops "
                    "being something to memorise.")
    def setup(self):
        self.plane = NumberPlane(
            x_range=[-2, 5, 1], y_range=[-1, 4, 1], x_length=5.6, y_length=4.3,
            background_line_style={"stroke_color": GREY_D, "stroke_width": 1,
                                   "stroke_opacity": 0.45},
            axis_config={"stroke_color": GREY_B, "stroke_width": 2,
                         "include_tip": False},
        ).move_to(ORIGIN_SHIFT + RIGHT * 1.6 + UP * 1.0)

        self.vec_a = self.arrow(self.a, A_C)
        self.vec_b = self.arrow(self.b, B_C)
        self.lab_a = MathTex(r"\vec{a}", color=A_C).next_to(self.pt(self.a), RIGHT, buff=0.18)
        self.lab_b = MathTex(r"\vec{b}", color=B_C).next_to(self.pt(self.b), UP, buff=0.15)
        self.title = Text("The dot product", font_size=30).to_edge(UP, buff=0.35)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), Create(self.plane), run_time=1.3)
        self.play(GrowArrow(self.vec_a), Write(self.lab_a), run_time=0.9)
        self.play(GrowArrow(self.vec_b), Write(self.lab_b), run_time=0.9)
        self.wait(0.4)

    @beat("Drop a perpendicular to cast a's shadow on b", seconds=11,
          narration="Shine a light straight down onto b, and a casts a shadow. "
                    "That shadow is the part of a that points along b — "
                    "everything perpendicular is discarded.")
    def project(self):
        p = self.a.projection_onto(self.b)          # computed, not eyeballed
        self.proj_vec = Arrow(ORIGIN_SHIFT, self.pt(p), buff=0, color=PROJ_C,
                              stroke_width=9, max_tip_length_to_length_ratio=0.22)
        self.proj_vec.set_z_index(self.vec_b.z_index - 1)   # sits under b
        self.drop = DashedLine(self.pt(self.a), self.pt(p), color=GREY_B,
                               stroke_width=2.5, dash_length=0.1)
        self.right_angle = RightAngle(
            Line(self.pt(p), self.pt(self.a)), Line(self.pt(p), ORIGIN_SHIFT),
            length=0.2, color=GREY_B, stroke_width=2)

        self.play(Create(self.drop), run_time=0.9)
        self.play(GrowArrow(self.proj_vec), Create(self.right_angle), run_time=1.0)

        shadow_len = self.a.scalar_projection(self.b)
        self.shadow_label = self.panel(
            Text("length of the shadow", font_size=22, color=GREY_B),
            MathTex(rf"{shadow_len:.2f}", font_size=44, color=PROJ_C),
            y=1.7)
        self.play(FadeIn(self.shadow_label, shift=UP * 0.15), run_time=0.8)
        self.wait(0.5)

    @beat("Shadow length times the length of b is the dot product", seconds=14,
          narration="Multiply that shadow by the length of b, and you get the "
                    "dot product. Not by convention — this is what the dot "
                    "product is. The arithmetic rule is a shortcut to this "
                    "number.")
    def identity(self):
        shadow = self.a.scalar_projection(self.b)
        dot = self.a.dot(self.b)
        assert dot_identity_check(self.a, self.b)     # checked before asserted

        eq = MathTex(
            rf"{shadow:.2f}", r"\times", rf"{self.b.norm:.2f}", "=",
            rf"{dot:.2f}", font_size=40,
        ).move_to([PANEL_X, 0.2, 0])
        eq[0].set_color(PROJ_C)
        eq[2].set_color(B_C)
        eq[4].set_color(WHITE)

        caption = Text("shadow × |b|", font_size=21, color=GREY_B)
        caption.next_to(eq, UP, buff=0.3)

        self.play(FadeIn(caption), run_time=0.5)
        self.play(Write(eq), run_time=1.4)
        self.wait(0.6)

        # The arithmetic rule, arriving at the same number.
        arith = MathTex(
            rf"{self.a.x:g}\cdot{self.b.x:g} + {self.a.y:g}\cdot{self.b.y:g}",
            "=", rf"{dot:.2f}", font_size=34,
        ).next_to(eq, DOWN, buff=0.55)
        arith[2].set_color(PROJ_C)
        self.play(Write(arith), run_time=1.2)
        self.play(Indicate(arith[2], color=PROJ_C, scale_factor=1.25), run_time=0.8)
        self.wait(0.8)
        self.play(FadeOut(caption), FadeOut(arith), FadeOut(self.shadow_label),
                  run_time=0.6)
        self.eq = eq

    @beat("Past ninety degrees the shadow flips and the sign goes negative", seconds=16,
          narration="Now swing a past ninety degrees. The shadow falls on the "
                    "other side of the origin, its length is negative, and so "
                    "is the dot product. The sign was never arbitrary — it is "
                    "telling you which way the vectors lean.")
    def obtuse(self):
        """Swing `a` past ninety degrees and watch the sign flip.

        The old arrow is faded out and a new one built from ``new_a``, rather
        than rotating or transforming the existing one. Rotate preserves
        length, so rotating by the angle difference left an arrow of |a|
        pointing along new_a — the picture and the arithmetic came from
        different vectors, which is the exact failure this project exists to
        prevent. Transform additionally left the original on the stage.
        """
        # Swings downward rather than straight back. (-3.4, -0.4) was very
        # nearly antiparallel to a, so old and new lay on one line and read as
        # a single double-headed arrow — and a 180-degree flip illustrates
        # nothing about crossing ninety degrees.
        new_a = Vec2(1.4, -2.2)
        p = new_a.projection_onto(self.b)
        dot = new_a.dot(self.b)
        # The beat's whole claim is that this is negative. Assert it rather
        # than trust it: the earlier vector sat at 45 degrees to b and the
        # scene confidently labelled a positive number "negative".
        assert dot < 0, f"beat claims a negative dot product, got {dot}"

        fresh_a = self.arrow(new_a, A_C)
        fresh_lab = MathTex(r"\vec{a}", color=A_C).next_to(self.pt(new_a), DR, buff=0.14)
        neg_proj = Arrow(ORIGIN_SHIFT, self.pt(p), buff=0, color=NEG_C,
                         stroke_width=9, max_tip_length_to_length_ratio=0.22)
        neg_proj.set_z_index(self.vec_b.z_index - 1)
        fresh_drop = DashedLine(self.pt(new_a), self.pt(p), color=GREY_B,
                                stroke_width=2.5, dash_length=0.1)

        # Clear every arrow in a's colour rather than only the tracked
        # reference. Animations can leave a copy on the stage that the
        # reference no longer points at, and a stale arrow is not cosmetic
        # here: two vectors labelled `a` at once make the scene unreadable.
        stale = [m for m in self.mobjects
                 if isinstance(m, (Arrow, MathTex))
                 and m is not self.eq
                 and m.get_color() == ManimColor(A_C)]
        self.play(FadeOut(self.right_angle), FadeOut(self.drop),
                  *[FadeOut(m) for m in stale], run_time=0.6)
        self.remove(*stale, self.drop, self.right_angle)

        self.vec_a, self.lab_a, self.drop = fresh_a, fresh_lab, fresh_drop
        self.play(GrowArrow(self.vec_a), Write(self.lab_a), run_time=1.0)
        self.play(ReplacementTransform(self.proj_vec, neg_proj),
                  Create(self.drop), run_time=1.0)
        self.proj_vec = neg_proj

        neg = MathTex(rf"{dot:.2f}", font_size=48, color=NEG_C).move_to([PANEL_X, 0.2, 0])
        note = Text("negative", font_size=24, color=NEG_C).next_to(neg, DOWN, buff=0.3)
        self.play(ReplacementTransform(self.eq, neg), run_time=1.0)
        self.eq = neg
        self.play(FadeIn(note, shift=UP * 0.15), run_time=0.6)
        self.wait(1.0)
        self.play(FadeOut(note), run_time=0.4)

    @beat("What the sign tells you", seconds=11,
          narration="Positive means the vectors broadly agree. Zero means they "
                    "are perpendicular — no shadow at all. Negative means they "
                    "oppose. One number, and it carries the whole relationship.")
    def summary(self):
        rows = VGroup(
            VGroup(Text("positive", font_size=24, color=PROJ_C),
                   Text("pointing together", font_size=20, color=GREY_B)),
            VGroup(Text("zero", font_size=24, color=GREY_B),
                   Text("perpendicular", font_size=20, color=GREY_B)),
            VGroup(Text("negative", font_size=24, color=NEG_C),
                   Text("pointing apart", font_size=20, color=GREY_B)),
        )
        for r in rows:
            r.arrange(RIGHT, buff=0.3, aligned_edge=DOWN)
        rows.arrange(DOWN, buff=0.34, aligned_edge=LEFT).move_to([PANEL_X, 0.3, 0])

        self.play(FadeOut(self.eq), FadeOut(self.drop), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows],
                              lag_ratio=0.3, run_time=1.8))
        self.wait(1.2)
