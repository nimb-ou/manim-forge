"""Gold scene 032 — rates of change, nested.

Calculus, Tier 2. The chain rule is easy to state and easy to misapply, and
the misapplication is always the same: evaluating the outer derivative at x
instead of at the value the inner function currently has.

So the scene shows that error rather than warning against it. At x = 0.4 the
correct answer is minus one point seven seven and the mistake gives plus two
point seven six -- not merely wrong in size but pointing the other way, which
is what makes it worth a beat. A worked example that is only slightly wrong
teaches nothing.

Every rate comes from ``chain_trace``, and ``verify_chain_rule`` checks the
product against a numeric derivative of the composite before the scene names
it.
"""

import math

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.calculus import (chain_trace, numeric_derivative,
                                       verify_chain_rule)

INNER_C = "#F0AC5F"
OUTER_C = BLUE_C
PROD_C = "#5CD0B3"
BAD_C = "#FC6255"
DIM = GREY_B

X0 = 0.4
K = 3.0


def inner(x):
    return K * x + 1.0


def inner_prime(x):
    return K


outer = math.sin
outer_prime = math.cos


class ChainRule(ForgeScene):

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 3.6).align_to(UP * 2.3, UP)

    @beat("Two machines, one feeding the other", seconds=17,
          narration="Two operations in a row. The first takes x and gives "
                    "three x plus one. The second takes that answer and gives "
                    "its sine. Feeding one into the other makes a single "
                    "function, and the question is how fast that combined "
                    "function changes.")
    def machines(self):
        self.title = Text("Rates of change, nested",
                          font_size=30).to_edge(UP, buff=0.4)

        boxes = VGroup()
        for lab, col in ((r"3x + 1", INNER_C), (r"\sin(\;)", OUTER_C)):
            b = VGroup(
                RoundedRectangle(width=2.1, height=1.15, corner_radius=0.12,
                                 color=col, stroke_width=2.6,
                                 fill_color=col, fill_opacity=0.12),
                MathTex(lab, font_size=28, color=col))
            b[1].move_to(b[0])
            boxes.add(b)
        boxes.arrange(RIGHT, buff=1.5).move_to(LEFT * 2.4 + UP * 1.1)

        arrows = VGroup(
            Arrow(LEFT * 5.6 + UP * 1.1, boxes[0].get_left(), buff=0.12,
                  color=DIM, stroke_width=2.4, max_tip_length_to_length_ratio=0.2),
            Arrow(boxes[0].get_right(), boxes[1].get_left(), buff=0.12,
                  color=DIM, stroke_width=2.4, max_tip_length_to_length_ratio=0.2),
            Arrow(boxes[1].get_right(), boxes[1].get_right() + RIGHT * 0.9,
                  buff=0.12, color=DIM, stroke_width=2.4,
                  max_tip_length_to_length_ratio=0.2),
        )
        tags = VGroup(
            MathTex("x", font_size=24, color=DIM).next_to(arrows[0], UP, buff=0.1),
            MathTex("u", font_size=24, color=INNER_C).next_to(arrows[1], UP, buff=0.1),
            MathTex(r"\sin u", font_size=24, color=OUTER_C).next_to(
                arrows[2], UP, buff=0.1),
        )

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(b, scale=0.8) for b in boxes],
                              lag_ratio=0.3, run_time=1.3))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows],
                              lag_ratio=0.2, run_time=1.1))
        self.play(FadeIn(tags), run_time=0.7)

        q = self.panel(MathTex(r"\frac{d}{dx}\sin(3x+1) = ?", font_size=30,
                               color=PROD_C))
        self.play(Write(q), run_time=1.2)
        self.boxes, self.arrows, self.tags, self.q = boxes, arrows, tags, q
        self.wait(1.2)

    @beat("Push a number through and watch both rates", seconds=27,
          narration="Take x equal to nought point four. The inner machine "
                    "turns it into two point two, and it does so three times "
                    "as fast as x moves — nudge x by a little and u moves "
                    "three times that little. Now the outer machine. Its rate "
                    "is the cosine, but the cosine of what? Not of x. Of two "
                    "point two, because that is the number actually arriving.")
    def trace(self):
        g, f, prod = chain_trace(inner, inner_prime, outer, outer_prime, X0)
        assert abs(g.value - 2.2) < 1e-9        # spoken
        assert g.rate == 3.0

        self.play(FadeOut(self.q), run_time=0.3)
        vals = VGroup(
            MathTex(rf"x = {X0}", font_size=26, color=DIM),
            MathTex(rf"u = {g.value:.1f}", font_size=26, color=INNER_C),
            MathTex(rf"\sin u = {f.value:.4f}", font_size=24, color=OUTER_C),
        ).arrange(DOWN, buff=0.26, aligned_edge=LEFT)
        vals.move_to(RIGHT * 3.6 + UP * 1.9)

        for v in vals:
            self.play(Write(v), run_time=0.7)

        rates = VGroup(
            VGroup(MathTex(r"\frac{du}{dx}", font_size=28, color=INNER_C),
                   MathTex(rf"= {g.rate:.0f}", font_size=26, color=INNER_C)
                   ).arrange(RIGHT, buff=0.2),
            VGroup(MathTex(r"\cos u", font_size=28, color=OUTER_C),
                   MathTex(rf"= {f.rate:.4f}", font_size=24, color=OUTER_C)
                   ).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        rates.next_to(vals, DOWN, buff=0.55).align_to(vals, LEFT)

        self.play(FadeIn(rates[0], shift=LEFT * 0.2), run_time=0.8)
        self.play(Indicate(self.boxes[0], color=INNER_C, scale_factor=1.06),
                  run_time=0.7)
        self.play(FadeIn(rates[1], shift=LEFT * 0.2), run_time=0.8)
        self.play(Indicate(self.boxes[1], color=OUTER_C, scale_factor=1.06),
                  run_time=0.7)
        self.vals, self.rates = vals, rates
        self.link = (g, f, prod)
        self.wait(1.4)

    @beat("Multiply the rates, and check it", seconds=20,
          narration="Multiply them. Three, times the cosine of two point two, "
                    "gives minus one point seven seven. That is the chain "
                    "rule: the rate of the whole is the rate of the inside "
                    "times the rate of the outside, taken where the inside "
                    "lands. Measuring the combined function directly gives the "
                    "same number.")
    def multiply(self):
        g, f, prod = self.link
        assert verify_chain_rule(inner, inner_prime, outer, outer_prime,
                                 [-1, 0, X0, 1, 2])
        num = numeric_derivative(lambda t: outer(inner(t)), X0)
        assert abs(prod - num) < 1e-5
        assert round(prod, 2) == -1.77          # spoken

        product = VGroup(
            MathTex(rf"{g.rate:.0f} \times {f.rate:.4f}", font_size=28,
                    color=PROD_C),
            MathTex(rf"= {prod:.4f}", font_size=32, color=PROD_C),
        ).arrange(DOWN, buff=0.24, aligned_edge=LEFT)
        product.next_to(self.rates, DOWN, buff=0.55).align_to(self.rates, LEFT)
        self.play(Write(product[0]), run_time=1.1)
        self.play(FadeIn(product[1], shift=UP * 0.12), run_time=0.8)

        check = VGroup(
            Text("measured directly", font_size=18, color=DIM),
            MathTex(rf"{num:.4f}", font_size=24, color=PROD_C),
        ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN)
        check.next_to(product, DOWN, buff=0.4).align_to(product, LEFT)
        self.play(FadeIn(check), run_time=0.9)
        self.product, self.check = product, check
        self.wait(1.6)

    @beat("The mistake everyone makes, and what it costs", seconds=26,
          narration="Here is the error worth seeing. Take the cosine of x "
                    "rather than of u — cosine of nought point four rather "
                    "than of two point two — and multiply by three. That "
                    "gives plus two point seven six. Not slightly off: the "
                    "wrong sign. The function is falling and the answer says "
                    "it is rising, because the outer rate was read at the "
                    "wrong place.")
    def mistake(self):
        g, f, prod = self.link
        wrong = outer_prime(X0) * inner_prime(X0)
        assert round(wrong, 2) == 2.76          # spoken
        assert wrong * prod < 0                 # genuinely the opposite sign

        self.play(FadeOut(self.vals), FadeOut(self.rates), FadeOut(self.check),
                  run_time=0.5)
        self.play(self.product.animate.scale(0.9).move_to(
            RIGHT * 3.6 + UP * 1.9), run_time=0.7)

        bad = VGroup(
            Text("reading cos at x, not u", font_size=19, color=BAD_C),
            MathTex(rf"\cos({X0}) \times {g.rate:.0f}", font_size=26, color=BAD_C),
            MathTex(rf"= {wrong:+.4f}", font_size=30, color=BAD_C),
        ).arrange(DOWN, buff=0.24, aligned_edge=LEFT)
        bad.next_to(self.product, DOWN, buff=0.65).align_to(self.product, LEFT)

        self.play(FadeIn(bad[0]), run_time=0.7)
        self.play(Write(bad[1]), run_time=1.0)
        self.play(FadeIn(bad[2], shift=UP * 0.12), run_time=0.8)
        self.play(Indicate(bad[2], color=BAD_C, scale_factor=1.15), run_time=0.9)

        verdict = VGroup(
            MathTex(rf"{prod:+.2f}", font_size=28, color=PROD_C),
            Text("vs", font_size=18, color=DIM),
            MathTex(rf"{wrong:+.2f}", font_size=28, color=BAD_C),
        ).arrange(RIGHT, buff=0.3)
        note = Text("opposite sign", font_size=21, color=BAD_C)
        tail = VGroup(verdict, note).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        tail.next_to(bad, DOWN, buff=0.5).align_to(bad, LEFT)
        self.play(FadeIn(tail, shift=UP * 0.12), run_time=1.0)
        self.wait(2.4)
