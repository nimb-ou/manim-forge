"""Gold scene 005 — Riemann sums converging to an integral.

Every rectangle drawn comes from ``forge.primitives.calculus``, every total is
summed from those rectangles, and the target is a reference integral computed
independently by fine midpoint summation. The scene can therefore show the
error shrinking honestly — and would show it failing to shrink if the maths
were wrong.

The chosen cubic integrates to exactly 8 over [0, 4], which gives the
convergence a clean target rather than a number that merely looks converged.

The contrast at the end is real rather than rhetorical: left-rule error halves
each time n doubles, midpoint error falls by four. Those rates are printed by
the primitive and are what the scene animates.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.calculus import RiemannSum, exact_integral

CURVE_C = BLUE_C
RECT_C = "#F0AC5F"
MID_C = "#5CD0B3"
DIM = GREY_B

A, B = 0.0, 4.0
PANEL_X = 3.9


def f(x: float) -> float:
    return 0.25 * x ** 3 - 1.2 * x ** 2 + 1.4 * x + 1.6


class RiemannConvergence(ForgeScene):

    exact = exact_integral(f, A, B)

    def build_axes(self) -> Axes:
        return Axes(
            x_range=[0, 4.4, 1], y_range=[0, 3.2, 1],
            x_length=6.4, y_length=4.4,
            axis_config={"include_tip": False, "stroke_width": 2,
                         "color": GREY_B, "font_size": 18},
        ).to_edge(LEFT, buff=0.75).shift(DOWN * 0.3)

    def bars(self, n: int, rule: str, color) -> VGroup:
        """Rectangles straight from the primitive — positions and heights are
        the ones the sum actually uses."""
        rs = RiemannSum(f, A, B, n, rule)
        g = VGroup()
        for r in rs.rectangles():
            if r.height <= 0:
                continue
            bl = self.axes.c2p(r.x_left, 0)
            tr = self.axes.c2p(r.x_right, r.height)
            g.add(Rectangle(width=tr[0] - bl[0], height=tr[1] - bl[1],
                            stroke_color=color, stroke_width=1.4,
                            fill_color=color, fill_opacity=0.35)
                  .move_to((bl + tr) / 2))
        return g

    def readout(self, n: int, total: float, err: float, color) -> VGroup:
        return VGroup(
            VGroup(Text("n =", font_size=24, color=DIM),
                   Text(str(n), font_size=34, color=color)
                   ).arrange(RIGHT, buff=0.22, aligned_edge=DOWN),
            MathTex(rf"\approx {total:.3f}", font_size=40, color=color),
            Text(f"off by {err:.3f}", font_size=22, color=DIM),
        ).arrange(DOWN, buff=0.22).move_to([PANEL_X, 0.9, 0])

    # -- beats ---------------------------------------------------------------

    @beat("Draw the curve and shade the area we want", seconds=6,
          narration="Here is a curve, and the area beneath it between zero and "
                    "four. That area is a single number — but there is no "
                    "obvious way to measure it, because the top edge keeps "
                    "curving.")
    def setup(self):
        self.axes = self.build_axes()
        self.curve = self.axes.plot(f, x_range=[A, B], color=CURVE_C, stroke_width=4)
        self.area = self.axes.get_area(self.curve, x_range=(A, B),
                                       color=CURVE_C, opacity=0.18)
        self.title = Text("Area under a curve", font_size=30).to_edge(UP, buff=0.35)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), Create(self.axes), run_time=1.2)
        self.play(Create(self.curve), run_time=1.6)
        self.play(FadeIn(self.area), run_time=1.1)
        self.wait(0.6)

    @beat("Approximate it crudely with two rectangles", seconds=7,
          narration="So approximate it with something we can measure. Two "
                    "rectangles, each as tall as the curve at its left edge. "
                    "It is clearly too small — every rectangle falls short "
                    "where the curve rises above it.")
    def crude(self):
        n = 2
        rs = RiemannSum(f, A, B, n, "left")
        self.rects = self.bars(n, "left", RECT_C)
        self.panel = self.readout(n, rs.total, abs(rs.total - self.exact), RECT_C)

        self.play(LaggedStart(*[FadeIn(r, scale=0.8) for r in self.rects],
                              lag_ratio=0.2, run_time=1.4))
        self.play(FadeIn(self.panel, shift=UP * 0.15), run_time=0.9)
        self.wait(1.2)

    @beat("Double the count and watch the gap close", seconds=12,
          narration="Now use more, thinner rectangles. Four. Eight. Sixteen. "
                    "Thirty-two. Each time the count doubles, the gap left "
                    "over is cut roughly in half — and the total climbs "
                    "steadily toward something.")
    def refine(self):
        for n in (4, 8, 16, 32):
            rs = RiemannSum(f, A, B, n, "left")
            fresh = self.bars(n, "left", RECT_C)
            panel = self.readout(n, rs.total, abs(rs.total - self.exact), RECT_C)
            # Fade rather than Transform: morphing one group of rectangles into
            # a differently-sized group pairs them arbitrarily and smears.
            self.play(FadeOut(self.rects, run_time=0.35))
            self.rects = fresh
            self.play(LaggedStart(*[FadeIn(r) for r in self.rects],
                                  lag_ratio=0.02, run_time=0.8),
                      ReplacementTransform(self.panel, panel))
            self.panel = panel
            self.wait(0.5)
        self.wait(0.6)

    @beat("Name the limit it is climbing towards", seconds=7,
          narration="That something is the integral: the exact area, equal to "
                    "eight. The rectangles never reach it with any finite "
                    "count — but they get as close as you care to ask.")
    def limit(self):
        target = VGroup(
            MathTex(r"\int_0^4 f(x)\,dx", font_size=44, color=CURVE_C),
            MathTex(rf"= {self.exact:.0f}", font_size=48, color=CURVE_C),
        ).arrange(DOWN, buff=0.28).move_to([PANEL_X, -1.5, 0])

        self.play(Write(target[0]), run_time=1.2)
        self.play(Write(target[1]), run_time=0.9)
        self.play(Indicate(target[1], color=CURVE_C, scale_factor=1.15), run_time=0.9)
        self.target = target
        self.wait(1.0)

    @beat("Sampling the middle instead converges far faster", seconds=10,
          narration="One change makes this dramatically better. Take each "
                    "rectangle's height from the middle of its interval rather "
                    "than the left edge. The overshoot on one side now cancels "
                    "the undershoot on the other — and with just eight "
                    "rectangles the error is smaller than the left rule "
                    "manages with sixty-four.")
    def midpoint(self):
        n = 8
        left = RiemannSum(f, A, B, n, "left")
        mid = RiemannSum(f, A, B, n, "midpoint")
        far_left = RiemannSum(f, A, B, 64, "left")
        # The scene's claim, asserted rather than trusted.
        assert abs(mid.total - self.exact) < abs(far_left.total - self.exact)

        fresh = self.bars(n, "midpoint", MID_C)
        self.play(FadeOut(self.rects, run_time=0.4))
        self.rects = fresh
        self.play(LaggedStart(*[FadeIn(r, scale=0.85) for r in self.rects],
                              lag_ratio=0.06, run_time=1.3))

        compare = VGroup(
            Text(f"left, n=8      off by {abs(left.total-self.exact):.3f}",
                 font_size=21, color=RECT_C),
            Text(f"left, n=64     off by {abs(far_left.total-self.exact):.3f}",
                 font_size=21, color=RECT_C),
            Text(f"midpoint, n=8  off by {abs(mid.total-self.exact):.3f}",
                 font_size=21, color=MID_C),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to([PANEL_X, 0.9, 0])

        self.play(FadeOut(self.panel), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.15) for c in compare],
                              lag_ratio=0.3, run_time=1.6))
        self.play(Indicate(compare[2], color=MID_C, scale_factor=1.1), run_time=0.9)
        self.wait(1.8)
