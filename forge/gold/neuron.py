"""Gold scene 036 — what a single neuron actually does.

Machine learning, Tier 2. A neuron is drawn so often as a circle with arrows
that people learn the picture without the arithmetic. It does exactly three
things: multiply each input by a weight, add them up with a bias, and bend the
result through a function that is not a straight line.

The third step is the one that matters, and the scene proves it rather than
asserting it: stack two linear neurons and the composition is shown to be a
single linear neuron, by computing the collapsed weights and checking they
agree everywhere. Without a bend, depth buys nothing at all.
"""

import math

from manim import *

from forge.beats import ForgeScene, beat

IN_C = "#F0AC5F"
W_C = BLUE_C
SUM_C = "#8B7FD4"
ACT_C = "#5CD0B3"
BAD_C = "#FC6255"
DIM = GREY_B

INPUTS = [0.9, -0.4, 0.7]
WEIGHTS = [1.2, -2.0, 0.5]
BIAS = 0.3


def relu(z):
    return max(0.0, z)


def weighted_sum(xs, ws, b):
    return sum(x * w for x, w in zip(xs, ws)) + b


class Neuron(ForgeScene):

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.0).align_to(UP * 2.3, UP)

    @beat("Three inputs, three weights", seconds=20,
          narration="A neuron with three inputs. Each input arrives with a "
                    "number attached to it — a weight — and the weight says "
                    "how much that input matters, and whether it counts for or "
                    "against. Nought point nine arriving on a weight of one "
                    "point two contributes one point zero eight.")
    def inputs(self):
        self.title = Text("What one neuron does", font_size=30).to_edge(UP, buff=0.4)

        self.body = Circle(radius=0.55, color=SUM_C, stroke_width=3,
                           fill_color=SUM_C, fill_opacity=0.16
                           ).move_to(LEFT * 1.2)
        ys = [1.5, 0.0, -1.5]
        self.in_dots, self.edges, self.in_labs, self.w_labs = [], [], VGroup(), VGroup()
        for i, (x, w, y) in enumerate(zip(INPUTS, WEIGHTS, ys)):
            src = np.array([-4.8, y, 0.0])
            d = Dot(src, color=IN_C, radius=0.09)
            e = Line(src, self.body.get_left(), color=W_C, stroke_width=2.4,
                     stroke_opacity=0.8)
            self.in_dots.append(d)
            self.edges.append(e)
            self.in_labs.add(MathTex(rf"{x}", font_size=24, color=IN_C
                                     ).next_to(d, LEFT, buff=0.18))
            self.w_labs.add(MathTex(rf"{w}", font_size=21, color=W_C
                                    ).move_to(e.point_from_proportion(0.45)
                                              + UP * 0.24))

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(FadeIn(self.body, scale=0.6), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in self.in_dots],
                              lag_ratio=0.2, run_time=1.0))
        self.play(FadeIn(self.in_labs), run_time=0.6)
        self.play(LaggedStart(*[Create(e) for e in self.edges],
                              lag_ratio=0.2, run_time=1.2))
        self.play(FadeIn(self.w_labs), run_time=0.7)

        first = INPUTS[0] * WEIGHTS[0]
        assert abs(first - 1.08) < 1e-9          # spoken in the narration
        note = self.panel(
            VGroup(MathTex(rf"{INPUTS[0]} \times {WEIGHTS[0]}", font_size=24,
                           color=IN_C),
                   MathTex(rf"= {first:.2f}", font_size=24, color=W_C)
                   ).arrange(RIGHT, buff=0.2),
        )
        self.play(Write(note), run_time=1.0)
        self.note = note
        self.wait(1.2)

    @beat("Add them up, with a bias", seconds=20,
          narration="Add the three contributions together, then add one more "
                    "number that belongs to the neuron itself rather than to "
                    "any input — the bias. It shifts the whole total up or "
                    "down, which is what lets the neuron fire readily or "
                    "reluctantly. Here the total comes to two point five "
                    "three.")
    def summation(self):
        z = weighted_sum(INPUTS, WEIGHTS, BIAS)
        assert abs(z - 2.53) < 1e-9              # spoken
        self.play(FadeOut(self.note), run_time=0.3)

        terms = VGroup(*[
            MathTex(rf"{x} \times {w} = {x*w:+.2f}", font_size=22, color=W_C)
            for x, w in zip(INPUTS, WEIGHTS)])
        terms.add(MathTex(rf"\text{{bias}} = {BIAS:+.2f}", font_size=22, color=DIM))
        terms.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        terms.move_to(RIGHT * 4.0 + UP * 1.7)

        self.play(LaggedStart(*[FadeIn(t, shift=LEFT * 0.2) for t in terms],
                              lag_ratio=0.3, run_time=2.0))
        total = VGroup(
            MathTex(r"z", font_size=28, color=SUM_C),
            MathTex(rf"= {z:.2f}", font_size=30, color=SUM_C),
        ).arrange(RIGHT, buff=0.2)
        total.next_to(terms, DOWN, buff=0.45).align_to(terms, LEFT)
        self.play(Write(total), run_time=1.0)
        self.play(Flash(self.body, color=SUM_C, line_length=0.18,
                        flash_radius=0.7, num_lines=12), run_time=0.9)
        self.terms, self.total, self.z = terms, total, z
        self.wait(1.2)

    @beat("Bend it, or the depth is wasted", seconds=18,
          narration="Then bend the total through a function with a kink in it. "
                    "This one passes positives through untouched and flattens "
                    "everything negative to zero. It looks almost too simple "
                    "to matter. It is the only reason stacking neurons "
                    "achieves anything, and here is why.")
    def activation(self):
        self.play(FadeOut(self.terms), FadeOut(self.total), run_time=0.4)
        axes = Axes(x_range=[-3, 3, 1], y_range=[-0.6, 3, 1],
                    x_length=3.6, y_length=2.6,
                    axis_config={"include_tip": False, "stroke_color": GREY_D,
                                 "stroke_width": 1.6, "font_size": 15},
                    ).move_to(RIGHT * 3.9 + UP * 0.9)
        curve = axes.plot(relu, x_range=[-3, 3, 0.02], color=ACT_C,
                          stroke_width=4)
        lab = MathTex(r"\max(0, z)", font_size=24, color=ACT_C).next_to(
            axes, UP, buff=0.2)

        self.play(Create(axes), run_time=0.9)
        self.play(Create(curve), Write(lab), run_time=1.4)

        a = relu(self.z)
        mark = Dot(axes.c2p(self.z, a), color=SUM_C, radius=0.08)
        drop = DashedLine(axes.c2p(self.z, 0), axes.c2p(self.z, a),
                          color=SUM_C, stroke_width=1.8, dash_length=0.07)
        self.play(Create(drop), FadeIn(mark, scale=0.5), run_time=0.9)

        out = VGroup(
            VGroup(MathTex(rf"z = {self.z:.2f}", font_size=24, color=SUM_C)),
            VGroup(MathTex(rf"\text{{out}} = {a:.2f}", font_size=26, color=ACT_C)),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        out.next_to(axes, DOWN, buff=0.5)
        self.play(FadeIn(out, shift=UP * 0.12), run_time=0.9)
        self.act_block = VGroup(axes, curve, lab, mark, drop, out)
        self.wait(1.4)

    @beat("Without the bend, two layers are one layer", seconds=27,
          narration="Take away the kink and stack two neurons. The first "
                    "multiplies by two and adds one; the second multiplies by "
                    "three and subtracts four. Work it through and the pair "
                    "is a single neuron that multiplies by six and subtracts "
                    "one — checked at every input, not just the one shown. "
                    "Any depth of straight lines collapses to one straight "
                    "line. The bend is what stops that happening.")
    def collapse(self):
        w1, b1 = 2.0, 1.0
        w2, b2 = 3.0, -4.0
        # Composition of two affine maps is affine, with these coefficients.
        wc, bc = w2 * w1, w2 * b1 + b2
        assert (wc, bc) == (6.0, -1.0)           # spoken
        xs = [-2.0, -0.5, 0.0, 0.9, 3.3]
        assert all(abs(w2 * (w1 * x + b1) + b2 - (wc * x + bc)) < 1e-12
                   for x in xs)

        self.play(FadeOut(self.act_block), FadeOut(self.in_labs),
                  FadeOut(self.w_labs), run_time=0.5)
        self.play(*[FadeOut(d) for d in self.in_dots],
                  *[FadeOut(e) for e in self.edges],
                  FadeOut(self.body), run_time=0.5)

        def cell(text, colour):
            box = RoundedRectangle(width=2.3, height=1.05, corner_radius=0.12,
                                   color=colour, stroke_width=2.6,
                                   fill_color=colour, fill_opacity=0.12)
            t = MathTex(text, font_size=24, color=colour).move_to(box)
            return VGroup(box, t)

        stacked = VGroup(cell(rf"{w1:.0f}x + {b1:.0f}", W_C),
                         cell(rf"{w2:.0f}u {b2:+.0f}", W_C)
                         ).arrange(RIGHT, buff=1.1).move_to(UP * 0.9)
        arrow = Arrow(stacked[0].get_right(), stacked[1].get_left(), buff=0.1,
                      color=DIM, stroke_width=2.4,
                      max_tip_length_to_length_ratio=0.22)
        self.play(FadeIn(stacked[0], scale=0.8), run_time=0.7)
        self.play(GrowArrow(arrow), FadeIn(stacked[1], scale=0.8), run_time=0.9)

        eq = MathTex(rf"{w2:.0f}({w1:.0f}x + {b1:.0f}) {b2:+.0f}",
                     r"=", rf"{wc:.0f}x {bc:+.0f}", font_size=32)
        eq[2].set_color(BAD_C)
        eq.move_to(DOWN * 0.7)
        self.play(Write(eq), run_time=1.6)

        one = cell(rf"{wc:.0f}x {bc:+.0f}", BAD_C).move_to(DOWN * 2.1)
        self.play(FadeIn(one, scale=0.8), run_time=0.9)

        verdict = Text("two layers, one line", font_size=22, color=BAD_C
                       ).next_to(one, RIGHT, buff=0.7)
        self.play(FadeIn(verdict, shift=LEFT * 0.15), run_time=0.9)
        self.wait(2.2)
