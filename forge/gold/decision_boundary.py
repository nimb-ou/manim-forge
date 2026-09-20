"""Gold scene 034 — a line that learns where to sit.

Machine learning, Tier 2. The perceptron is the smallest honest example of
learning: it does nothing while it is right, and moves only when it is wrong.
That character is the whole lesson, and it is exactly what gets lost if the
boundary is animated smoothly from its starting position to its final one.

So the boundary here jumps. ``Perceptron.fit`` records the line after every
correction and the scene steps through that history, sitting still through the
points it already gets right.

The second half is the part usually left out. Widen the clouds until they
overlap and the perceptron never settles -- not slowly, never. It is still
correcting after five hundred passes. That is not a tuning problem; a straight
line cannot separate those points, and the algorithm has no way to say so.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.learning import Perceptron, blobs, is_separable

POS_C = "#5CD0B3"
NEG_C = "#F0AC5F"
LINE_C = BLUE_C
BAD_C = "#FC6255"
DIM = GREY_B

RATE = 0.03
EPOCHS = 60


class DecisionBoundary(ForgeScene):

    def make_axes(self):
        return Axes(x_range=[-3.4, 3.4, 1], y_range=[-2.6, 2.6, 1],
                    x_length=6.6, y_length=4.6,
                    axis_config={"include_tip": False, "stroke_color": GREY_D,
                                 "stroke_width": 1.6, "font_size": 16},
                    ).shift(LEFT * 2.2 + DOWN * 0.25)

    def boundary(self, w, b, colour=LINE_C, width=3.5):
        """The line w.x + b = 0, clipped to the axes."""
        if abs(w[1]) < 1e-9:
            x = -b / w[0] if abs(w[0]) > 1e-9 else 0.0
            return Line(self.axes.c2p(x, -2.5), self.axes.c2p(x, 2.5),
                        color=colour, stroke_width=width)
        ys = [(-b - w[0] * x) / w[1] for x in (-3.3, 3.3)]
        ys = [max(min(y, 2.5), -2.5) for y in ys]
        xs = [-3.3, 3.3]
        return Line(self.axes.c2p(xs[0], ys[0]), self.axes.c2p(xs[1], ys[1]),
                    color=colour, stroke_width=width)

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.2).align_to(UP * 2.3, UP)

    @beat("Two kinds of thing, and a line to tell them apart", seconds=15,
          narration="Thirty-six examples of two kinds. The task is to find a "
                    "straight line with one kind on each side. Nobody hands "
                    "the machine the line; it starts with a guess that is "
                    "barely a guess, and corrects.")
    def data(self):
        self.title = Text("A line that learns where to sit",
                          font_size=30).to_edge(UP, buff=0.4)
        self.axes = self.make_axes()
        self.pts, self.labs = blobs()
        assert is_separable(self.pts, self.labs)

        self.dots = VGroup(*[
            Dot(self.axes.c2p(*p), color=POS_C if y == 1 else NEG_C,
                radius=0.075)
            for p, y in zip(self.pts, self.labs)])

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.axes), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in self.dots],
                              lag_ratio=0.04, run_time=1.8))

        key = self.panel(
            VGroup(Dot(color=POS_C, radius=0.08),
                   Text("class A", font_size=20, color=POS_C)
                   ).arrange(RIGHT, buff=0.25),
            VGroup(Dot(color=NEG_C, radius=0.08),
                   Text("class B", font_size=20, color=NEG_C)
                   ).arrange(RIGHT, buff=0.25),
            Text("find a line between them", font_size=19, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(k, shift=LEFT * 0.2) for k in key],
                              lag_ratio=0.3, run_time=1.5))
        self.key = key
        self.wait(1.2)

    @beat("It moves only when it is wrong", seconds=24,
          narration="Here is the rule, and it is the whole algorithm. Take one "
                    "example. If the line already puts it on the right side, "
                    "do nothing at all. If it does not, tilt the line a little "
                    "towards that example. Nothing else. Watch: the line sits "
                    "perfectly still through every point it already handles, "
                    "and jumps only at the ones it gets wrong.")
    def train(self):
        self.play(FadeOut(self.key), run_time=0.3)
        p = Perceptron()
        fixes = p.fit(self.pts, self.labs, epochs=EPOCHS, rate=RATE)
        assert fixes == 6                       # spoken in the next beat
        assert p.accuracy(self.pts, self.labs) == 1.0

        counter = VGroup(
            MathTex("0", font_size=34, color=LINE_C),
            Text("corrections", font_size=19, color=DIM),
        ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN)
        counter.move_to(RIGHT * 4.2 + UP * 2.0)
        self.play(FadeIn(counter), run_time=0.5)

        line = None
        for i, (w, b) in enumerate(p.history):
            if all(abs(v) < 1e-12 for v in w) and abs(b) < 1e-12:
                continue                     # the zero start has no line to draw
            new = self.boundary(w, b)
            if line is None:
                self.play(Create(new), run_time=0.9)
            else:
                self.play(Transform(line, new), run_time=0.8)
                self.wait(0.45)              # the stillness is the point
            line = line or new
            c = MathTex(str(i), font_size=34, color=LINE_C).move_to(counter[0])
            self.play(FadeTransform(counter[0], c), run_time=0.2)
            counter.submobjects[0] = c

        self.line, self.counter = line, counter
        self.model, self.fixes = p, fixes
        self.wait(1.2)

    @beat("Six corrections, and it never needs another", seconds=24,
          narration="Six corrections out of thirty-six examples, and after "
                    "that the line is right about every one of them, so the "
                    "rule stops changing it. That is not luck. If some line "
                    "separates the data, this procedure is guaranteed to find "
                    "one — that is the perceptron convergence theorem, and it "
                    "is why such a simple rule was worth taking seriously.")
    def converged(self):
        acc = self.model.accuracy(self.pts, self.labs)
        assert acc == 1.0 and self.fixes == 6

        halo = VGroup(*[
            Circle(radius=0.13, color=POS_C if y == 1 else NEG_C,
                   stroke_width=2).move_to(self.axes.c2p(*pt))
            for pt, y in zip(self.pts, self.labs)])
        self.play(LaggedStart(*[FadeIn(h, scale=0.6) for h in halo],
                              lag_ratio=0.03, run_time=1.6))

        res = self.panel(
            VGroup(MathTex(rf"{self.fixes}", font_size=34, color=LINE_C),
                   Text("corrections", font_size=19, color=DIM)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(MathTex(r"100\%", font_size=30, color=POS_C),
                   Text("correct", font_size=19, color=DIM)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            Text("and it stops moving", font_size=19, color=DIM),
        )
        self.play(FadeOut(self.counter), run_time=0.3)
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in res],
                              lag_ratio=0.3, run_time=1.5))
        self.halo, self.res = halo, res
        self.wait(1.4)

    @beat("Overlap the classes and it never stops", seconds=25,
          narration="Now move the two groups together until they overlap. Run "
                    "exactly the same rule. It corrects, and corrects, and "
                    "has made over five hundred corrections by the end of "
                    "sixty passes, never settling anywhere. No straight line "
                    "separates these "
                    "points, so there is nothing for it to converge to — and "
                    "the algorithm has no way to tell you that. It just keeps "
                    "going.")
    def overlap(self):
        opts, olabs = blobs(36, seed=0, sep=0.8, spread=0.85)
        assert not is_separable(opts, olabs)
        q = Perceptron()
        g = q.fit(opts, olabs, epochs=EPOCHS, rate=RATE)
        assert g > 500 and q.accuracy(opts, olabs) < 1.0

        self.play(FadeOut(self.halo), FadeOut(self.res), FadeOut(self.dots),
                  FadeOut(self.line), run_time=0.6)
        new_dots = VGroup(*[
            Dot(self.axes.c2p(*p), color=POS_C if y == 1 else NEG_C, radius=0.075)
            for p, y in zip(opts, olabs)])
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in new_dots],
                              lag_ratio=0.04, run_time=1.6))

        # A dozen boundaries spread across the run: it is still moving at the end.
        picks = [q.history[i] for i in
                 range(1, len(q.history), max(1, len(q.history) // 12))][:12]
        line = self.boundary(*picks[0], colour=BAD_C)
        self.play(Create(line), run_time=0.7)
        for w, b in picks[1:]:
            self.play(Transform(line, self.boundary(w, b, colour=BAD_C)),
                      run_time=0.34)

        res = self.panel(
            VGroup(MathTex(rf"{g}", font_size=34, color=BAD_C),
                   Text("corrections", font_size=19, color=DIM)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(MathTex(rf"{q.accuracy(opts, olabs):.0%}".replace("%", r"\%"),
                           font_size=30, color=BAD_C),
                   Text("correct", font_size=19, color=DIM)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            Text("still moving", font_size=20, color=BAD_C),
            Text("no line can separate them", font_size=18, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in res],
                              lag_ratio=0.3, run_time=1.8))
        self.wait(2.2)
