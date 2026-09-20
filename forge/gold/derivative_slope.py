"""Gold scene 021 — the derivative as a limit of slopes.

Calculus, Tier 2, and the scene the rest of calculus leans on. The move that
matters is the one most explanations skip: the secant is a slope you can
actually compute, for a gap you can actually see, and the derivative is what
those computed numbers settle on. Drawing a tangent line and calling it the
derivative asserts the answer and shows none of the argument.

So the secants are real. ``secant_sequence`` computes the slope at each gap
and the scene displays those numbers as they arrive; the tangent is drawn only
once the sequence has visibly settled, and its slope is checked against a
numeric derivative before the scene names it.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.calculus import (numeric_derivative, secant_sequence,
                                       verify_derivative)

CURVE_C = BLUE_C
SEC_C = "#F0AC5F"
TAN_C = "#5CD0B3"
DIM = GREY_B

X0 = 1.0
GAPS = [1.5, 1.0, 0.6, 0.35, 0.2, 0.1, 0.05]


def f(x):
    return 0.45 * x ** 3 - 1.1 * x ** 2 + 0.6 * x + 1.7


def fprime(x):
    return 1.35 * x ** 2 - 2.2 * x + 0.6


class DerivativeSlope(ForgeScene):

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.3).align_to(UP * 2.3, UP)

    @beat("A curve, and one point on it", seconds=20,
          narration="A curve, and a point sitting on it. The question is how "
                    "steep the curve is right there — not nearby, not on "
                    "average, but at that exact point. Steepness is rise over "
                    "run, and at a single point there is no run to divide by. "
                    "That is the whole difficulty.")
    def setup(self):
        self.title = Text("The derivative as a limit of slopes",
                          font_size=29).to_edge(UP, buff=0.4)

        self.axes = Axes(
            x_range=[-0.6, 3.2, 1], y_range=[0.4, 3.4, 1],
            x_length=7.0, y_length=4.6,
            axis_config={"include_tip": False, "stroke_color": GREY_B,
                         "stroke_width": 2, "font_size": 20},
        ).shift(LEFT * 1.7 + DOWN * 0.5)
        self.curve = self.axes.plot(f, x_range=[-0.5, 3.1, 0.02],
                                    color=CURVE_C, stroke_width=4)
        self.p = Dot(self.axes.c2p(X0, f(X0)), color=WHITE, radius=0.09)
        plab = MathTex(rf"x = {X0:.0f}", font_size=24, color=DIM).next_to(
            self.p, DL, buff=0.16)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.axes), run_time=1.3)
        self.play(Create(self.curve), run_time=2.0)
        self.play(FadeIn(self.p, scale=0.4), FadeIn(plab), run_time=0.8)
        self.plab = plab
        self.wait(1.4)

    @beat("A second point gives a slope you can compute", seconds=22,
          narration="Put a second point on the curve, some distance along. Now "
                    "there is a run, so there is a slope: the rise between "
                    "them divided by the gap. That is a real number, and it is "
                    "not the answer — it is the average steepness across the "
                    "whole stretch, which is not what was asked.")
    def secant(self):
        h0 = GAPS[0]
        self.h = ValueTracker(h0)

        def q_point():
            h = self.h.get_value()
            return self.axes.c2p(X0 + h, f(X0 + h))

        self.q = always_redraw(lambda: Dot(q_point(), color=SEC_C, radius=0.08))
        self.line = always_redraw(lambda: Line(
            *self._extend(self.axes.c2p(X0, f(X0)), q_point()),
            color=SEC_C, stroke_width=3.4))
        self.rise = always_redraw(lambda: DashedLine(
            self.axes.c2p(X0 + self.h.get_value(), f(X0)),
            q_point(), color=DIM, stroke_width=2, dash_length=0.08))
        self.run = always_redraw(lambda: DashedLine(
            self.axes.c2p(X0, f(X0)),
            self.axes.c2p(X0 + self.h.get_value(), f(X0)),
            color=DIM, stroke_width=2, dash_length=0.08))

        self.play(FadeIn(self.q, scale=0.5), run_time=0.6)
        self.play(Create(self.run), Create(self.rise), run_time=1.0)
        self.play(Create(self.line), run_time=1.0)

        self.readout = self.panel(
            always_redraw(lambda: VGroup(
                Text("gap", font_size=20, color=DIM),
                MathTex(rf"{self.h.get_value():.2f}", font_size=30, color=SEC_C),
            ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN).move_to(
                RIGHT * 4.3 + UP * 2.0)),
            always_redraw(lambda: VGroup(
                Text("slope", font_size=20, color=DIM),
                MathTex(rf"{self._slope():.3f}", font_size=30, color=SEC_C),
            ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN).move_to(
                RIGHT * 4.3 + UP * 1.35)),
        )
        self.play(FadeIn(self.readout), run_time=0.9)
        self.wait(1.6)

    def _slope(self):
        h = self.h.get_value()
        return (f(X0 + h) - f(X0)) / h

    def _extend(self, a, b, k=1.5):
        d = b - a
        return a - d * (k - 1) * 0.35, b + d * (k - 1)

    @beat("Shrink the gap and watch the number settle", seconds=22,
          narration="Now close the gap. The second point slides towards the "
                    "first and the slope changes as it goes — but it does not "
                    "wander. It settles. One point one four, then nought "
                    "point four five, then closer and closer to a single "
                    "value it never quite reaches, because reaching it would "
                    "mean dividing by zero.")
    def shrink(self):
        seq = secant_sequence(f, X0, GAPS)
        # The narration speaks the first two aloud, so they are pinned
        # here: a changed curve or gap list must break the scene loudly
        # rather than leave the voice track quoting stale numbers.
        assert abs(seq[0][1] - 1.14) < 0.005, seq[0]
        assert abs(seq[1][1] - 0.45) < 0.005, seq[1]

        rows = VGroup(*[
            VGroup(MathTex(rf"{h:.2f}", font_size=22, color=DIM),
                   MathTex(r"\to", font_size=20, color=DIM),
                   MathTex(rf"{s:.3f}", font_size=24, color=SEC_C),
                   ).arrange(RIGHT, buff=0.22)
            for h, s in seq])
        rows.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        rows.move_to(RIGHT * 4.3 + DOWN * 0.7)

        for i, (h, _) in enumerate(seq[1:], start=1):
            self.play(self.h.animate.set_value(h), run_time=0.85)
            self.play(FadeIn(rows[i - 1], shift=LEFT * 0.15), run_time=0.3)
        self.play(FadeIn(rows[len(seq) - 1], shift=LEFT * 0.15), run_time=0.3)
        self.seq_rows = rows
        self.wait(1.4)

    @beat("The value it settles on is the derivative", seconds=22,
          narration="That settling value is the derivative. Here it is minus "
                    "zero point two five, and the line through the point with "
                    "that slope is the tangent — the straight line that "
                    "matches the curve's direction exactly. The formula gives "
                    "the same number at every point, which is checked against "
                    "the slopes themselves before it is written down.")
    def tangent(self):
        d = fprime(X0)
        assert abs(d - (-0.25)) < 1e-9        # spoken in the narration
        assert verify_derivative(f, fprime, [0.0, 0.5, 1.0, 2.0, 3.0])
        assert abs(d - numeric_derivative(f, X0)) < 1e-6

        self.play(FadeOut(self.q), FadeOut(self.line), FadeOut(self.rise),
                  FadeOut(self.run), FadeOut(self.readout), run_time=0.7)

        p0 = self.axes.c2p(X0, f(X0))
        dx = 1.3
        tan = Line(self.axes.c2p(X0 - dx, f(X0) - d * dx),
                   self.axes.c2p(X0 + dx, f(X0) + d * dx),
                   color=TAN_C, stroke_width=4.5)
        self.play(Create(tan), run_time=1.4)

        answer = VGroup(
            MathTex(r"f'(x) = 1.35x^2 - 2.2x + 0.6", font_size=26, color=TAN_C),
            VGroup(MathTex(rf"f'({X0:.0f})", font_size=28, color=DIM),
                   MathTex(rf"= {d:.3f}", font_size=30, color=TAN_C)
                   ).arrange(RIGHT, buff=0.2, aligned_edge=DOWN),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        answer.move_to(RIGHT * 4.0 + UP * 1.9)

        self.play(FadeOut(self.seq_rows), run_time=0.4)
        self.play(Write(answer[0]), run_time=1.4)
        self.play(FadeIn(answer[1], shift=UP * 0.12), run_time=0.8)
        self.wait(2.2)
