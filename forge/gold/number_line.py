"""Gold scene 009 — the number line and what lives on it.

Tier 1. Nothing here is difficult, and that is the point: every later scene
assumes a number line has been drawn well, and a model that has never seen one
done properly will draw one badly inside a scene about something else.

The density argument at the end is computed rather than asserted — midpoints
are taken repeatedly and the gap is reported shrinking, so "there is always
another one between" is demonstrated rather than claimed.
"""

from fractions import Fraction
import math

from manim import *

from forge.beats import ForgeScene, beat

INT_C = BLUE_C
FRAC_C = "#F0AC5F"
IRR_C = "#FC6255"
DIM = GREY_B


class NumberLine1D(ForgeScene):

    @beat("Draw the line and mark the whole numbers", seconds=17,
          narration="A line, and a point on it we agree to call zero. Step the "
                    "same distance right, again and again, and you have the "
                    "whole numbers. Step left instead and you have the "
                    "negatives. Everything else in mathematics gets placed "
                    "against this line.")
    def integers(self):
        self.line = NumberLine(x_range=[-4, 4, 1], length=9.0, include_numbers=True,
                               font_size=24, color=GREY_B,
                               decimal_number_config={"num_decimal_places": 0})
        self.line.shift(UP * 0.4)
        self.title = Text("What lives on the number line?",
                          font_size=30).to_edge(UP, buff=0.4)

        zero = Dot(self.line.n2p(0), color=INT_C, radius=0.08)
        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.line), run_time=1.8)
        self.play(FadeIn(zero, scale=0.4), run_time=0.5)
        self.play(LaggedStart(*[
            Flash(self.line.n2p(k), color=INT_C, line_length=0.12,
                  flash_radius=0.18, num_lines=8)
            for k in range(-4, 5) if k != 0], lag_ratio=0.12, run_time=2.0))
        self.wait(0.6)

    @beat("Fill the gaps with fractions", seconds=12,
          narration="Between any two whole numbers there is space, and "
                    "fractions fill it. Halves, then thirds, then fifths — and "
                    "however fine you go, there is still room between them.")
    def fractions(self):
        marks = VGroup()
        for den in (2, 4):
            row = VGroup()
            for num in range(-4 * den, 4 * den + 1):
                f = Fraction(num, den)
                if f.denominator != den:        # already drawn at a coarser step
                    continue
                row.add(Line(self.line.n2p(float(f)) + DOWN * 0.11,
                             self.line.n2p(float(f)) + UP * 0.11,
                             color=FRAC_C, stroke_width=2).set_opacity(0.8))
            marks.add(row)
            self.play(LaggedStart(*[Create(m) for m in row],
                                  lag_ratio=0.01, run_time=1.2))
        label = MathTex(r"\tfrac{1}{2},\ \tfrac{1}{4},\ \tfrac{1}{8},\ \dots",
                        font_size=34, color=FRAC_C).next_to(self.line, DOWN, buff=0.9)
        self.play(Write(label), run_time=0.9)
        self.marks, self.frac_label = marks, label
        self.wait(0.8)

    @beat("Some points are no fraction at all", seconds=15,
          narration="But not every point is a fraction. The diagonal of a unit "
                    "square lands here — root two — and no fraction, however "
                    "finely chosen, lands exactly on it. The line is fuller "
                    "than the fractions can account for.")
    def irrational(self):
        r2 = math.sqrt(2)
        sq = Square(side_length=self.line.n2p(1)[0] - self.line.n2p(0)[0],
                    color=IRR_C, stroke_width=3, fill_opacity=0.12)
        sq.move_to(self.line.n2p(0.5) + UP * 0.75, aligned_edge=DOWN)
        diag = Line(sq.get_corner(DL), sq.get_corner(UR), color=IRR_C, stroke_width=3)

        self.play(FadeOut(self.frac_label), run_time=0.3)
        self.play(Create(sq), run_time=0.9)
        self.play(Create(diag), run_time=0.8)

        landed = diag.copy().set_color(IRR_C)
        dot = Dot(self.line.n2p(r2), color=IRR_C, radius=0.08)
        lab = MathTex(r"\sqrt{2}", font_size=32, color=IRR_C).next_to(dot, DOWN, buff=0.25)
        self.play(Rotate(landed, -PI / 4, about_point=sq.get_corner(DL)), run_time=1.2)
        self.play(FadeOut(landed), FadeIn(dot, scale=0.5), Write(lab), run_time=0.8)
        self.play(FadeOut(sq), FadeOut(diag), run_time=0.5)
        self.sq_group = VGroup(dot, lab)
        self.wait(0.9)

    @beat("Between any two points there is always another", seconds=19,
          narration="And here is the strange part. Take any two points, however "
                    "close. Their midpoint sits between them. Take that "
                    "midpoint and one of the originals — there is another one "
                    "between those. You never run out of room. The line has no "
                    "gaps and no smallest step.")
    def density(self):
        a, b = 1.0, 2.0
        rows = VGroup()
        # The shrinking gap is computed, so "always another between" is
        # demonstrated rather than asserted.
        for i in range(5):
            mid = (a + b) / 2
            d = Dot(self.line.n2p(mid), color=INT_C, radius=0.06 - i * 0.007)
            self.play(FadeIn(d, scale=0.5), run_time=0.45)
            rows.add(Text(f"gap {b - a:.5f}", font_size=19, color=DIM))
            b = mid
        rows.arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        rows.move_to([4.2, -1.1, 0])
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.1) for r in rows],
                              lag_ratio=0.2, run_time=1.4))
        closing = VGroup(
            Text("never zero,", font_size=22, color=INT_C),
            Text("never the last one", font_size=22, color=INT_C),
        ).arrange(DOWN, buff=0.1).next_to(rows, DOWN, buff=0.35)
        self.play(FadeIn(closing, shift=UP * 0.12), run_time=0.8)
        self.wait(1.4)
