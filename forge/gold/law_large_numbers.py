"""Gold scene 008 — the law of large numbers.

Driven by ``forge.primitives.probability``. The flips are genuinely sampled
(seeded, so renders reproduce) and the curve drawn is the real running
proportion — which wanders, because real runs wander. A hand-drawn curve
settling smoothly onto one half would be a lie about randomness, and the
wandering is the thing worth showing.

The 1/sqrt(n) envelope is computed, not sketched, so the claim that the wander
shrinks at that rate is visible rather than asserted.
"""

import math

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.probability import CoinRun

RUN_C = "#F0AC5F"
HALF_C = "#5CD0B3"
ENV_C = "#9A72AC"
DIM = GREY_B

N = 600
PANEL_X = 4.3


class LawOfLargeNumbers(ForgeScene):

    run = CoinRun(n=N, p=0.5, seed=7)

    @beat("Flip a coin ten times and look at the proportion", seconds=14,
          narration="Flip a fair coin ten times. You might expect five heads. "
                    "This run gave seven — a proportion of zero point seven, "
                    "nowhere near a half. Small samples are not gently wrong. "
                    "They are wildly wrong.")
    def small(self):
        rp = self.run.running_proportion()
        self.axes = Axes(
            x_range=[0, N, N // 4], y_range=[0, 1, 0.25],
            x_length=7.0, y_length=4.0,
            axis_config={"include_tip": False, "stroke_width": 2,
                         "color": GREY_D, "font_size": 17},
        ).to_edge(LEFT, buff=0.8).shift(DOWN * 0.2)

        self.half = DashedLine(self.axes.c2p(0, 0.5), self.axes.c2p(N, 0.5),
                               color=HALF_C, stroke_width=2.5, dash_length=0.12)
        half_lab = Text("0.5", font_size=20, color=HALF_C).next_to(
            self.axes.c2p(N, 0.5), RIGHT, buff=0.12)
        self.title = Text("Small samples lie", font_size=30).to_edge(UP, buff=0.35)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), Create(self.axes), run_time=1.2)
        self.play(Create(self.half), FadeIn(half_lab), run_time=0.8)

        first = VMobject(stroke_color=RUN_C, stroke_width=4)
        first.set_points_smoothly([self.axes.c2p(i, v) for i, v in rp[:10]])
        self.play(Create(first), run_time=1.6)

        p10 = rp[9][1]
        note = VGroup(
            Text(f"{p10:.2f}", font_size=60, color=RUN_C),
            Text("after 10 flips", font_size=21, color=DIM),
        ).arrange(DOWN, buff=0.16).move_to([PANEL_X, 1.5, 0])
        self.play(FadeIn(note, shift=UP * 0.15), run_time=0.8)
        self.first, self.note = first, note
        self.wait(1.2)

    @beat("Keep flipping and watch it wander inward", seconds=16,
          narration="Keep going. It does not march politely to a half — it "
                    "wanders, overshooting and correcting. But the wandering "
                    "gets smaller. Not because later flips correct earlier "
                    "ones, but because each new flip counts for less against a "
                    "growing total.")
    def converge(self):
        rp = self.run.running_proportion()
        full = VMobject(stroke_color=RUN_C, stroke_width=3)
        full.set_points_smoothly([self.axes.c2p(i, v) for i, v in rp])

        new_title = Text("but large ones cannot", font_size=30).to_edge(UP, buff=0.35)
        self.play(FadeOut(self.first), run_time=0.3)
        self.play(Create(full), Transform(self.title, new_title),
                  run_time=5.0, rate_func=linear)
        self.full = full

        marks = VGroup()
        for k in (50, 200, 600):
            v = rp[k - 1][1]
            d = Dot(self.axes.c2p(k, v), color=RUN_C, radius=0.055)
            t = Text(f"{v:.3f}", font_size=17, color=RUN_C).next_to(d, UP, buff=0.1)
            marks.add(VGroup(d, t))
        self.play(FadeOut(self.note), run_time=0.3)
        self.play(LaggedStart(*[FadeIn(m) for m in marks], lag_ratio=0.3, run_time=1.4))
        self.marks = marks
        self.wait(1.0)

    @beat("The wander shrinks like one over root n", seconds=19,
          narration="And the shrinking has a shape. The typical distance from "
                    "a half falls like one over the square root of the number "
                    "of flips — so to halve your error you need four times the "
                    "data. That is why polls are expensive, and why they are "
                    "never quite certain.")
    def envelope(self):
        r = self.run
        upper = VMobject(stroke_color=ENV_C, stroke_width=2.5)
        lower = VMobject(stroke_color=ENV_C, stroke_width=2.5)
        ks = list(range(5, N + 1, 5))
        upper.set_points_smoothly([self.axes.c2p(k, 0.5 + r.standard_error(k)) for k in ks])
        lower.set_points_smoothly([self.axes.c2p(k, 0.5 - r.standard_error(k)) for k in ks])

        self.play(FadeOut(self.marks), run_time=0.4)
        self.play(Create(upper), Create(lower), run_time=1.8)

        rows = VGroup()
        for k in (10, 50, 200, 600):
            rows.add(Text(f"n={k:<4} off by {r.deviation_at(k):.3f}   "
                          f"envelope {r.standard_error(k):.3f}",
                          font_size=19, color=DIM))
        rows.arrange(DOWN, buff=0.18, aligned_edge=LEFT).move_to([PANEL_X, 1.0, 0])
        formula = MathTex(r"\sigma \sim \tfrac{1}{\sqrt{n}}", font_size=42, color=ENV_C)
        formula.next_to(rows, DOWN, buff=0.45)

        self.play(LaggedStart(*[FadeIn(x, shift=LEFT * 0.15) for x in rows],
                              lag_ratio=0.25, run_time=1.5))
        self.play(Write(formula), run_time=1.1)
        self.wait(1.8)
