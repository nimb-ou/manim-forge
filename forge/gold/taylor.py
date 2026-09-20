"""Gold scene 024 — approximating a curve with polynomials.

Calculus, Tier 2. Taylor series are usually presented as more terms, better
fit, forever. That is false, and the scene says so: the same polynomial that
tracks sine beautifully near zero diverges spectacularly a little further out,
and watching it fail is the only way the radius of convergence means anything.

Every coefficient is computed from the derivatives rather than written down,
so an added term cannot silently use the wrong factorial. Each derivative is
checked against a numeric one before it is used, and the claim that more terms
help is checked on the interval the scene draws -- and separately shown to
fail outside it.
"""

import math

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.calculus import (taylor_error, taylor_poly,
                                       verify_derivative,
                                       verify_taylor_improves)

CURVE_C = BLUE_C
APPROX_C = "#F0AC5F"
GOOD_C = "#5CD0B3"
BAD_C = "#FC6255"
DIM = GREY_B

#: sin and its derivatives, cycling with period four.
DERIVS = [math.sin, math.cos, lambda x: -math.sin(x), lambda x: -math.cos(x),
          math.sin, math.cos, lambda x: -math.sin(x), lambda x: -math.cos(x)]
ORDERS = [1, 3, 5, 7]
A = 0.0
TEST_X = 1.4


class Taylor(ForgeScene):

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.3).align_to(UP * 2.3, UP)

    def poly(self, order):
        """Taylor polynomial up to x^order, coefficients computed."""
        return taylor_poly(DERIVS[:order + 1], A)

    @beat("A curve, and a straight line that matches it at one point", seconds=18,
          narration="Here is the sine curve. Suppose all you are allowed is a "
                    "polynomial — additions and multiplications, nothing "
                    "else. How close can you get? Start with the simplest "
                    "thing that touches the curve at zero and leaves in the "
                    "same direction: the line y equals x.")
    def first(self):
        assert verify_derivative(math.sin, math.cos, [-2, -0.5, 0, 1, 2.5])

        self.title = Text("Approximating a curve with polynomials",
                          font_size=28).to_edge(UP, buff=0.4)
        self.axes = Axes(
            x_range=[-6.6, 6.6, 1], y_range=[-2.6, 2.6, 1],
            x_length=8.2, y_length=4.4,
            axis_config={"include_tip": False, "stroke_color": GREY_B,
                         "stroke_width": 2, "font_size": 18},
        ).shift(LEFT * 1.5 + DOWN * 0.45)
        self.curve = self.axes.plot(math.sin, x_range=[-6.5, 6.5, 0.03],
                                    color=CURVE_C, stroke_width=4)
        origin = Dot(self.axes.c2p(0, 0), color=WHITE, radius=0.075)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.axes), run_time=1.2)
        self.play(Create(self.curve), run_time=2.0)
        self.play(FadeIn(origin, scale=0.5), run_time=0.5)

        p1 = self.poly(1)
        self.approx = self.axes.plot(p1, x_range=[-6.5, 6.5, 0.03],
                                     color=APPROX_C, stroke_width=3.5)
        self.play(Create(self.approx), run_time=1.4)

        self.label = self.panel(
            MathTex("x", font_size=32, color=APPROX_C),
            Text("matches at 0, and its slope", font_size=18, color=DIM),
        )
        self.play(FadeIn(self.label), run_time=0.8)
        self.order_now = 1
        self.wait(1.4)

    @beat("Each extra term fixes the next derivative", seconds=20,
          narration="Add a term chosen so the polynomial's second derivative "
                    "matches too, then its third, then its fourth. Each new "
                    "coefficient is the matching derivative of sine at zero, "
                    "divided by that many factorial — computed, not looked "
                    "up. Watch the approximation grip the curve over a wider "
                    "and wider stretch each time.")
    def more_terms(self):
        rows = VGroup()
        for order in ORDERS:
            terms = []
            for k in range(order + 1):
                c = DERIVS[k](A) / math.factorial(k)
                if abs(c) < 1e-12:
                    continue
                sign = "-" if c < 0 else ("+" if terms else "")
                den = round(1 / abs(c))
                terms.append(rf"{sign}\frac{{x^{k}}}{{{den}}}" if k > 1
                             else rf"{sign}x")
            rows.add(MathTex("".join(terms), font_size=26, color=APPROX_C))
        rows.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        rows.move_to(RIGHT * 4.3 + UP * 1.2)

        self.play(FadeOut(self.label), run_time=0.3)
        self.play(FadeIn(rows[0]), run_time=0.6)

        for i, order in enumerate(ORDERS[1:], start=1):
            p = self.poly(order)
            new = self.axes.plot(p, x_range=[-6.5, 6.5, 0.03],
                                 color=APPROX_C, stroke_width=3.5)
            self.play(Transform(self.approx, new), run_time=1.4)
            self.play(FadeIn(rows[i], shift=LEFT * 0.2), run_time=0.5)
            self.order_now = order
            self.wait(0.4)
        self.term_rows = rows
        self.wait(1.0)

    @beat("Near the centre, the error collapses", seconds=22,
          narration="Measure it. At x equals one point four, the straight line "
                    "is out by more than four tenths. Three terms brings that "
                    "down to four hundredths, five terms to two thousandths, "
                    "seven terms to six parts in a hundred thousand. Each term "
                    "really does help — and the scene checks that it does "
                    "before saying so.")
    def errors(self):
        assert verify_taylor_improves(math.sin, DERIVS[:8], A, TEST_X)
        # The narration speaks all four aloud, so each is pinned to the
        # *phrase* used rather than to a number chosen after the fact.
        e = [abs(taylor_error(math.sin, DERIVS[:o + 1], A, TEST_X))
             for o in ORDERS]
        assert e[0] > 0.4                        # "more than four tenths"
        assert round(e[1], 2) == 0.04            # "four hundredths"
        assert round(e[2], 3) == 0.002           # "two thousandths"
        assert round(e[3] * 1e5) == 6            # "six parts in a hundred thousand"

        self.play(FadeOut(self.term_rows), run_time=0.4)
        marker = DashedLine(self.axes.c2p(TEST_X, -2.4),
                            self.axes.c2p(TEST_X, 2.4),
                            color=GOOD_C, stroke_width=2, dash_length=0.1)
        xlab = MathTex(rf"x = {TEST_X}", font_size=22, color=GOOD_C).next_to(
            self.axes.c2p(TEST_X, -2.4), DOWN, buff=0.12)
        self.play(Create(marker), FadeIn(xlab), run_time=1.0)

        rows = VGroup()
        for order in ORDERS:
            e = abs(taylor_error(math.sin, DERIVS[:order + 1], A, TEST_X))
            rows.add(VGroup(
                MathTex(rf"{order}", font_size=22, color=DIM),
                Text("terms", font_size=17, color=DIM),
                MathTex(rf"{e:.5f}", font_size=24, color=GOOD_C),
            ).arrange(RIGHT, buff=0.22))
        rows.arrange(DOWN, buff=0.24, aligned_edge=LEFT)
        rows.move_to(RIGHT * 4.3 + UP * 1.0)

        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.35, run_time=2.0))
        self.err_rows = VGroup(rows, marker, xlab)
        self.wait(1.6)

    @beat("Far from the centre, it falls apart", seconds=29,
          narration="But more terms is not always better, and this is where "
                    "most explanations stop too early. Follow the same "
                    "seven-term polynomial out past five and it does not "
                    "track sine at all — it dives away and never comes back. "
                    "The approximation is local. It is built entirely from "
                    "what sine does at zero, and far enough from zero that "
                    "information runs out. Adding terms widens the good "
                    "stretch; it does not make it everything.")
    def divergence(self):
        far = 5.6
        e_near = abs(taylor_error(math.sin, DERIVS[:8], A, TEST_X))
        e_far = abs(taylor_error(math.sin, DERIVS[:8], A, far))
        assert e_far > 1.0 > e_near          # the whole point of the beat
        # The "more terms always helps" rule genuinely fails out here.
        assert not verify_taylor_improves(math.sin, DERIVS[:8], A, far)

        self.play(FadeOut(self.err_rows), run_time=0.4)

        wide = self.axes.plot(self.poly(7), x_range=[-6.5, 6.5, 0.02],
                              color=BAD_C, stroke_width=3.5)
        self.play(Transform(self.approx, wide), run_time=1.2)

        band = Rectangle(width=self.axes.c2p(3.3, 0)[0] - self.axes.c2p(-3.3, 0)[0],
                         height=4.3, color=GOOD_C, stroke_width=2,
                         fill_color=GOOD_C, fill_opacity=0.07
                         ).move_to(self.axes.c2p(0, 0))
        band_lab = Text("good here", font_size=19, color=GOOD_C).next_to(
            band, UP, buff=0.12)
        self.play(FadeIn(band), FadeIn(band_lab), run_time=1.0)

        rows = self.panel(
            VGroup(MathTex(rf"x = {TEST_X}", font_size=24, color=DIM),
                   MathTex(rf"{e_near:.5f}", font_size=26, color=GOOD_C)
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
            VGroup(MathTex(rf"x = {far}", font_size=24, color=DIM),
                   MathTex(rf"{e_far:.1f}", font_size=26, color=BAD_C)
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
            Text("same polynomial", font_size=18, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.35, run_time=1.8))
        self.wait(2.6)
