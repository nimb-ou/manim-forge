"""Gold scene 035 — why one learning rate cannot suit two directions.

Machine learning, Tier 2, building on the one-dimensional learning-rate scene.
On a bowl, every reasonable rate works and the lesson is invisible. The
interesting surface is a valley: steep across, shallow along. A step small
enough to be stable across the valley crawls along it, and a step large enough
to make progress along it explodes across it. That tension is why momentum,
Adam, and per-parameter rates exist at all.

Every path is produced by running the update rule. ``stable_rate`` is a real
property of the surface -- one over twice the largest curvature -- so "too
large" is checked against the surface rather than against how the picture
happens to look, and ``verify_rate_threshold`` confirms that a rate below it
converges and one well above it does not.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.learning import Descent2D, Surface2D, verify_rate_threshold

SLOW_C = BLUE_C
GOOD_C = "#5CD0B3"
BAD_C = "#FC6255"
CONTOUR_C = GREY_D
DIM = GREY_B

VALLEY = Surface2D(a=6.0, b=0.4)
START = (-1.6, 1.3)


class GradientDescent2D(ForgeScene):

    def make_axes(self):
        return Axes(x_range=[-2.2, 2.2, 1], y_range=[-1.8, 1.8, 1],
                    x_length=6.4, y_length=4.4,
                    axis_config={"include_tip": False, "stroke_color": GREY_D,
                                 "stroke_width": 1.6, "font_size": 16},
                    ).shift(LEFT * 2.2 + DOWN * 0.25)

    def contours(self):
        """Level sets of a x^2 + b y^2 -- ellipses, wide along the shallow axis."""
        g = VGroup()
        for level in (0.4, 1.2, 2.6, 4.6, 7.2):
            rx = (level / VALLEY.a) ** 0.5
            ry = (level / VALLEY.b) ** 0.5
            e = Ellipse(width=2 * rx * self.unit_x, height=2 * ry * self.unit_y,
                        color=CONTOUR_C, stroke_width=1.6,
                        stroke_opacity=0.8).move_to(self.axes.c2p(0, 0))
            g.add(e)
        return g

    def trail(self, path, colour, width=3.2):
        pts = [self.axes.c2p(max(min(x, 2.15), -2.15), max(min(y, 1.75), -1.75))
               for x, y in path]
        dots = VGroup(*[Dot(p, color=colour, radius=0.045) for p in pts])
        segs = VGroup(*[Line(a, b, color=colour, stroke_width=width,
                             stroke_opacity=0.8)
                        for a, b in zip(pts, pts[1:])])
        return VGroup(segs, dots)

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.3).align_to(UP * 2.3, UP)

    @beat("A valley, not a bowl", seconds=19,
          narration="A loss surface shaped like a valley: steep across, "
                    "gentle along. The rings are lines of equal loss, and they "
                    "are stretched — fifteen times more curved in one "
                    "direction than the other. Real problems look like this. "
                    "A perfectly round bowl is the case where nothing "
                    "interesting happens.")
    def surface(self):
        self.title = Text("One rate, two directions",
                          font_size=30).to_edge(UP, buff=0.4)
        self.axes = self.make_axes()
        self.unit_x = (self.axes.c2p(1, 0) - self.axes.c2p(0, 0))[0]
        self.unit_y = (self.axes.c2p(0, 1) - self.axes.c2p(0, 0))[1]
        assert round(VALLEY.condition) == 15        # spoken

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.axes), run_time=1.0)
        rings = self.contours()
        self.play(LaggedStart(*[Create(r) for r in rings],
                              lag_ratio=0.18, run_time=2.0))
        self.rings = rings

        self.start_dot = Dot(self.axes.c2p(*START), color=WHITE, radius=0.09)
        self.min_dot = Dot(self.axes.c2p(0, 0), color=GOOD_C, radius=0.07)
        self.play(FadeIn(self.start_dot, scale=0.5),
                  FadeIn(self.min_dot, scale=0.5), run_time=0.8)

        facts = self.panel(
            VGroup(Text("curvature ratio", font_size=19, color=DIM),
                   MathTex(rf"{VALLEY.condition:.0f}:1", font_size=26,
                           color=BAD_C)).arrange(RIGHT, buff=0.25,
                                                 aligned_edge=DOWN),
            Text("steep across", font_size=19, color=DIM),
            Text("gentle along", font_size=19, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(f, shift=LEFT * 0.2) for f in facts],
                              lag_ratio=0.3, run_time=1.6))
        self.facts = facts
        self.wait(1.2)

    @beat("A small step is safe, and painfully slow", seconds=22,
          narration="Take a step small enough to be safe across the valley. It "
                    "is safe — every step reduces the loss and the path never "
                    "wobbles. But look where it goes: straight down the steep "
                    "walls in a few steps, then a long, patient crawl along "
                    "the floor, sixty steps later still not at the bottom.")
    def slow(self):
        assert verify_rate_threshold(VALLEY)
        rate = VALLEY.stable_rate * 0.5
        d = Descent2D(VALLEY, rate, START)
        path = d.run(60)
        assert not d.diverged

        self.play(FadeOut(self.facts), run_time=0.3)
        tr = self.trail(path, SLOW_C)
        self.play(Create(tr[0]), run_time=2.2)
        self.play(LaggedStart(*[FadeIn(dot, scale=0.5) for dot in tr[1]],
                              lag_ratio=0.02, run_time=1.2))

        res = self.panel(
            VGroup(Text("rate", font_size=19, color=DIM),
                   MathTex(rf"{rate:.3f}", font_size=26, color=SLOW_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(Text("after 60 steps", font_size=19, color=DIM),
                   MathTex(rf"{d.final_loss:.3f}", font_size=26, color=SLOW_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            Text("safe, but crawling", font_size=19, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in res],
                              lag_ratio=0.3, run_time=1.5))
        self.slow_trail, self.slow_res = tr, res
        self.slow_loss = d.final_loss
        self.wait(1.4)

    @beat("A larger step explodes across the valley", seconds=20,
          narration="So take a bigger step. Two and a half times the stable "
                    "rate. Along the floor that would be a real improvement — "
                    "but across the valley it overshoots, lands further up the "
                    "opposite wall than it started, overshoots again, and the "
                    "loss runs away to something astronomical in thirty "
                    "steps.")
    def fast(self):
        rate = VALLEY.stable_rate * 2.5
        d = Descent2D(VALLEY, rate, START)
        path = d.run(60)
        assert d.diverged
        assert d.final_loss > 1e6

        self.play(FadeOut(self.slow_trail), FadeOut(self.slow_res),
                  run_time=0.5)
        tr = self.trail(path[:14], BAD_C, width=3.4)
        self.play(Create(tr[0]), run_time=2.0)
        self.play(LaggedStart(*[FadeIn(dot, scale=0.5) for dot in tr[1]],
                              lag_ratio=0.06, run_time=1.0))

        res = self.panel(
            VGroup(Text("rate", font_size=19, color=DIM),
                   MathTex(rf"{rate:.3f}", font_size=26, color=BAD_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(Text("after 30 steps", font_size=19, color=DIM),
                   MathTex(rf"{d.final_loss:.1e}".replace("e+", r"\times 10^{")
                           + "}", font_size=22, color=BAD_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            Text("diverged", font_size=20, color=BAD_C),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in res],
                              lag_ratio=0.3, run_time=1.5))
        self.fast_trail, self.fast_res = tr, res
        self.wait(1.4)

    @beat("The two demands cannot both be met", seconds=25,
          narration="That is the bind. The largest stable step is set by the "
                    "steepest direction, and the progress you need is set by "
                    "the shallowest. With curvatures fifteen to one apart, any "
                    "single number is either too big for one direction or too "
                    "small for the other. Every optimiser worth its name — "
                    "momentum, per-parameter rates, Adam — exists to stop "
                    "using one number for both.")
    def bind(self):
        t = VALLEY.stable_rate
        assert abs(t - 1.0 / (2 * VALLEY.a)) < 1e-12

        self.play(FadeOut(self.fast_trail), FadeOut(self.fast_res), run_time=0.5)
        mid = Descent2D(VALLEY, t * 0.95, START)
        mid.run(60)
        tr = self.trail(mid.path, GOOD_C)
        self.play(Create(tr[0]), run_time=1.6)

        rows = VGroup(
            VGroup(Text("stable limit", font_size=19, color=DIM),
                   MathTex(rf"{t:.3f}", font_size=26, color=GOOD_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(Text("set by", font_size=19, color=DIM),
                   MathTex(r"\frac{1}{2 \cdot \text{max curvature}}",
                           font_size=24, color=GOOD_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(Text("progress set by", font_size=19, color=DIM),
                   Text("min curvature", font_size=20, color=SLOW_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
        ).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        rows.move_to(RIGHT * 4.3 + UP * 1.6)

        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.3, run_time=1.8))
        verdict = VGroup(
            MathTex(rf"{VALLEY.condition:.0f}\times", font_size=32, color=BAD_C),
            Text("apart", font_size=20, color=DIM),
        ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN)
        note = Text("one number cannot serve both", font_size=19, color=BAD_C)
        tail = VGroup(verdict, note).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        tail.next_to(rows, DOWN, buff=0.55).align_to(rows, LEFT)
        self.play(FadeIn(tail, shift=UP * 0.12), run_time=1.2)
        self.wait(2.4)
