"""Gold scene 041 — why the bell curve keeps turning up.

Probability, Tier 2. Most demonstrations of the central limit theorem start
from data that is already vaguely bell-shaped, which proves nothing. This one
starts from a source with two separated humps and *no values at all* near the
middle -- and the averages pile up exactly where no single measurement ever
lands.

The quantitative half is the part a picture cannot carry. A histogram visibly
narrowing is obvious; whether it narrows at the right rate is the actual
claim. ``verify_spread_shrinks_as_sqrt_n`` measures the spread at n = 4, 16
and 64 and compares it against one over root n, and the scene shows those
measured numbers beside the prediction rather than asserting agreement.
"""

import math

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.probability import (histogram, sample_means, spread,
                                          verify_spread_shrinks_as_sqrt_n)

RAW_C = "#FC6255"
MEAN_C = "#5CD0B3"
FIT_C = "#F0AC5F"
DIM = GREY_B

KIND = "bimodal"
N_SAMPLES = 4000
BINS = 44
LO, HI = 0.0, 1.0
STAGES = [4, 16, 64]


class CentralLimit(ForgeScene):

    def make_axes(self, y_max):
        return Axes(x_range=[LO, HI, 0.25], y_range=[0, y_max, y_max],
                    x_length=7.0, y_length=3.5,
                    axis_config={"include_tip": False, "stroke_color": GREY_D,
                                 "stroke_width": 1.6, "font_size": 16},
                    y_axis_config={"stroke_opacity": 0.0},
                    ).move_to(LEFT * 1.6 + DOWN * 0.6)

    def bars(self, counts, colour, y_max):
        g = VGroup()
        w = (HI - LO) / BINS
        for i, c in enumerate(counts):
            if c == 0:
                continue
            x0 = LO + i * w
            top = self.axes.c2p(x0 + w / 2, min(c, y_max))
            base = self.axes.c2p(x0 + w / 2, 0)
            h = max(top[1] - base[1], 0.008)
            r = Rectangle(width=(self.axes.c2p(w, 0) - self.axes.c2p(0, 0))[0]
                          * 0.86, height=h, color=colour, fill_color=colour,
                          fill_opacity=0.75, stroke_width=0.6)
            r.move_to(np.array([base[0], base[1] + h / 2, 0.0]))
            g.add(r)
        return g

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.5).align_to(UP * 2.3, UP)

    @beat("A source with a hole in the middle", seconds=20,
          narration="Here is what we are drawing from, and it is deliberately "
                    "nothing like a bell. Two separated humps, with a gap "
                    "between them where no measurement ever falls. Values near "
                    "nought point five simply do not occur. Remember that, "
                    "because it is about to stop being true of the averages.")
    def source(self):
        self.title = Text("Why the bell curve keeps appearing",
                          font_size=29).to_edge(UP, buff=0.4)
        raw = sample_means(1, N_SAMPLES, seed=3, kind=KIND)
        counts = histogram(raw, BINS, LO, HI)
        self.y_max = max(counts) * 1.08
        self.axes = self.make_axes(self.y_max)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.axes), run_time=0.9)
        bars = self.bars(counts, RAW_C, self.y_max)
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.15) for b in bars],
                              lag_ratio=0.012, run_time=2.2))

        # The middle really is empty, not merely sparse.
        mid = [c for i, c in enumerate(counts)
               if 0.42 < LO + (i + 0.5) * (HI - LO) / BINS < 0.58]
        assert sum(mid) == 0, sum(mid)
        gap = Rectangle(width=(self.axes.c2p(0.16, 0) - self.axes.c2p(0, 0))[0],
                        height=2.2, color=DIM, stroke_width=1.6,
                        stroke_opacity=0.7).move_to(
            self.axes.c2p(0.5, self.y_max * 0.42))
        glab = Text("nothing here", font_size=17, color=DIM).next_to(
            gap, UP, buff=0.14)
        self.play(Create(gap), FadeIn(glab), run_time=1.0)

        self.raw_spread = spread(raw)
        note = self.panel(
            Text("one measurement", font_size=20, color=RAW_C),
            VGroup(Text("spread", font_size=19, color=DIM),
                   MathTex(rf"{self.raw_spread:.4f}", font_size=24, color=RAW_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
        )
        self.play(LaggedStart(*[FadeIn(n, shift=LEFT * 0.2) for n in note],
                              lag_ratio=0.3, run_time=1.3))
        self.bars_now, self.gap, self.glab, self.note = bars, gap, glab, note
        self.wait(1.4)

    @beat("Average four of them and the hole fills in", seconds=25,
          narration="Now take four measurements at a time and plot their "
                    "average. Straight away the two humps are gone and the "
                    "middle — where nothing was ever measured — is where most "
                    "of the averages land. To get an average near nought point "
                    "five you need roughly half your draws from each hump, and "
                    "that is much the commonest way for four draws to come "
                    "out.")
    def average_four(self):
        n = STAGES[0]
        vals = sample_means(n, N_SAMPLES, seed=3, kind=KIND)
        counts = histogram(vals, BINS, LO, HI)
        s = spread(vals)

        self.play(FadeOut(self.gap), FadeOut(self.glab), FadeOut(self.note),
                  run_time=0.4)
        new = self.bars(counts, MEAN_C, self.y_max)
        self.play(FadeOut(self.bars_now), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.15) for b in new],
                              lag_ratio=0.012, run_time=1.8))
        self.bars_now = new

        note = self.panel(
            VGroup(Text("average of", font_size=19, color=DIM),
                   MathTex(rf"{n}", font_size=26, color=MEAN_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(Text("spread", font_size=19, color=DIM),
                   MathTex(rf"{s:.4f}", font_size=24, color=MEAN_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            Text("humps gone", font_size=19, color=MEAN_C),
        )
        self.play(LaggedStart(*[FadeIn(x, shift=LEFT * 0.2) for x in note],
                              lag_ratio=0.3, run_time=1.4))
        self.note = note
        self.wait(1.4)

    @beat("More per average, and it tightens", seconds=17,
          narration="Sixteen at a time, then sixty-four. The shape does not "
                    "change any further — it is already the bell — but it "
                    "draws in around the true average. The source never "
                    "changed. Only how many of its measurements go into each "
                    "point being plotted.")
    def tighten(self):
        self.play(FadeOut(self.note), run_time=0.3)
        rows = VGroup()
        self.spreads = {}
        for n in STAGES:
            vals = sample_means(n, N_SAMPLES, seed=3, kind=KIND)
            self.spreads[n] = spread(vals)
            counts = histogram(vals, BINS, LO, HI)
            new = self.bars(counts, MEAN_C, self.y_max)
            self.play(FadeOut(self.bars_now), run_time=0.3)
            self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.12) for b in new],
                                  lag_ratio=0.012, run_time=1.2))
            self.bars_now = new
            rows.add(VGroup(
                MathTex(rf"n = {n}", font_size=22, color=DIM),
                MathTex(rf"{self.spreads[n]:.4f}", font_size=22, color=MEAN_C),
            ).arrange(RIGHT, buff=0.3))
            rows.arrange(DOWN, buff=0.24, aligned_edge=LEFT)
            rows.move_to(RIGHT * 4.5 + UP * 1.5)
            self.play(FadeIn(rows[-1], shift=LEFT * 0.2), run_time=0.4)
        self.rows = rows
        self.wait(1.2)

    @beat("And it tightens at a particular rate", seconds=26,
          narration="The narrowing is not vague. Quadruple how many go into "
                    "each average and the spread halves — it falls like one "
                    "over the square root of n. Here are the measured spreads "
                    "beside what that rule predicts, and they agree to within "
                    "about one percent at every step. That is the theorem: "
                    "whatever you started from, the averages become normal, "
                    "and they do it at this speed.")
    def rate(self):
        assert verify_spread_shrinks_as_sqrt_n(KIND)
        self.play(FadeOut(self.rows), run_time=0.3)

        head = VGroup(
            Text("measured", font_size=17, color=MEAN_C),
            Text("predicted", font_size=17, color=FIT_C),
            Text("off by", font_size=17, color=DIM),
        ).arrange(RIGHT, buff=0.42)
        table = VGroup(head)
        worst = 0.0
        for n in STAGES:
            got = self.spreads[n]
            want = self.raw_spread / math.sqrt(n)
            err = abs(got - want) / want
            worst = max(worst, err)
            table.add(VGroup(
                MathTex(rf"{got:.4f}", font_size=21, color=MEAN_C),
                MathTex(rf"{want:.4f}", font_size=21, color=FIT_C),
                MathTex(rf"{err:.1%}".replace("%", r"\%"), font_size=19,
                        color=DIM),
            ).arrange(RIGHT, buff=0.42))
        assert worst < 0.015                   # "about one percent"
        table.arrange(DOWN, buff=0.26, aligned_edge=LEFT)
        table.move_to(RIGHT * 4.3 + UP * 1.5)

        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in table],
                              lag_ratio=0.3, run_time=2.0))
        law = MathTex(r"\sigma_{\bar{x}} = \frac{\sigma}{\sqrt{n}}",
                      font_size=34, color=FIT_C).next_to(table, DOWN, buff=0.6)
        self.play(Write(law), run_time=1.4)

        tail = Text("whatever you started from", font_size=18,
                    color=DIM).next_to(law, DOWN, buff=0.45)
        self.play(FadeIn(tail), run_time=0.8)
        self.wait(2.4)
