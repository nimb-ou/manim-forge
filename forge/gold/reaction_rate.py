"""Gold scene 016 — concentration falling into an exponential.

Chemistry, Tier 2, and the one that ties chemistry to the calculus scenes.
The claim is that a constant per-particle chance of reacting produces an
exponential fall in concentration -- and the scene earns it rather than
drawing a smooth curve and labelling it.

Every particle in the box is simulated individually: each surviving one gets
an independent chance to react on each step. The dots that vanish on screen
are the ones the simulation killed, the plotted points are the counts that
simulation produced, and the smooth exponential is drawn over the top as a
prediction. They agree because the model is right, not because the same
function drew both.

``verify_decay_matches_exponential`` checks the agreement against sampling
uncertainty rather than a fixed percentage. In the tail, where a handful of
particles remain, a fixed tolerance measures how far along the axis you looked
rather than whether the model is correct.
"""

import random

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.chem import simulate_decay, verify_decay_matches_exponential

PART_C = "#5CD0B3"
GONE_C = "#FC6255"
CURVE_C = "#F0AC5F"
DIM = GREY_B

N0 = 150
P_REACT = 0.06
STEPS = 45
BOX_W, BOX_H = 4.3, 3.5
BOX_CENTRE = LEFT * 3.5 + DOWN * 0.35


class ReactionRate(ForgeScene):

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.3).align_to(UP * 2.3, UP)

    @beat("A box of particles, each with the same chance", seconds=15,
          narration="A box of reactant. Every particle in it is identical, and "
                    "on any given moment each one has the same small chance of "
                    "reacting — independent of the others, and independent of "
                    "how long it has already been waiting.")
    def setup_box(self):
        self.title = Text("Why concentration falls exponentially",
                          font_size=29).to_edge(UP, buff=0.4)

        self.box = Rectangle(width=BOX_W, height=BOX_H, color=GREY_C,
                             stroke_width=2).move_to(BOX_CENTRE)

        rng = random.Random(11)          # layout only; the physics is elsewhere
        self.particles = VGroup(*[
            Dot(BOX_CENTRE + np.array([rng.uniform(-BOX_W / 2 + 0.12,
                                                   BOX_W / 2 - 0.12),
                                       rng.uniform(-BOX_H / 2 + 0.12,
                                                   BOX_H / 2 - 0.12), 0.0]),
                color=PART_C, radius=0.055)
            for _ in range(N0)])

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.box), run_time=0.9)
        self.play(LaggedStart(*[FadeIn(self.particles[i::6], scale=0.6)
                                for i in range(6)],
                              lag_ratio=0.25, run_time=1.8))

        self.rule = self.panel(
            VGroup(Text("start", font_size=20, color=DIM),
                   MathTex(rf"{N0}", font_size=30, color=PART_C)
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
            VGroup(Text("chance per step", font_size=20, color=DIM),
                   MathTex(rf"{P_REACT:.2f}", font_size=30, color=CURVE_C)
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in self.rule],
                              lag_ratio=0.35, run_time=1.2))
        self.wait(0.9)

    @beat("Run it, and watch them go", seconds=17,
          narration="Run the clock. Early on a great many react, simply "
                    "because there are a great many left to react. Later, "
                    "fewer and fewer go — not because any particle has become "
                    "less likely to react, but because there are fewer of them "
                    "to try.")
    def run_sim(self):
        # The simulation is the source of truth. The dots that disappear are
        # exactly the ones it killed, in the order it killed them.
        self.run = simulate_decay(n0=N0, p=P_REACT, steps=STEPS, dt=1.0, seed=3)
        counts = self.run.counts

        self.counter = MathTex(rf"{N0}", font_size=46, color=PART_C).next_to(
            self.box, DOWN, buff=0.35)
        self.play(FadeIn(self.counter), run_time=0.5)

        alive = list(range(N0))
        rng = random.Random(5)
        rng.shuffle(alive)               # which dot dies is cosmetic; how many is not

        frames = [(4, 8), (8, 14), (14, 22), (22, 32), (32, STEPS)]
        for lo, hi in frames:
            dead = counts[lo] - counts[hi]
            victims = [self.particles[alive.pop()] for _ in range(dead)]
            new_count = MathTex(rf"{counts[hi]}", font_size=46,
                                color=PART_C).move_to(self.counter)
            self.play(
                LaggedStart(*[v.animate.set_color(GONE_C).set_opacity(0.0)
                              for v in victims],
                            lag_ratio=0.008, run_time=1.5),
                FadeTransform(self.counter, new_count),
            )
            self.counter = new_count
        self.wait(0.9)

    @beat("Plot the count against time", seconds=19,
          narration="Plot what just happened. The points are the counts from "
                    "the simulation — nothing smoothed, nothing fitted. The "
                    "curve over them is the prediction: the starting amount "
                    "times e to the minus k t. They lie on top of each other, "
                    "which is the claim being made good.")
    def plot(self):
        assert verify_decay_matches_exponential()

        self.play(FadeOut(self.box), FadeOut(self.particles),
                  FadeOut(self.counter), FadeOut(self.rule), run_time=0.8)

        self.axes = Axes(
            x_range=[0, STEPS, 10], y_range=[0, N0 * 1.08, 50],
            x_length=8.4, y_length=4.2,
            axis_config={"include_tip": False, "stroke_color": GREY_B,
                         "stroke_width": 2, "font_size": 20},
        ).shift(DOWN * 0.55)
        xl = self.axes.get_x_axis_label(Text("time", font_size=19, color=DIM),
                                        edge=DOWN, direction=DOWN, buff=0.3)
        yl = self.axes.get_y_axis_label(
            Text("particles left", font_size=19, color=DIM), edge=UP,
            direction=UP, buff=0.2)
        self.play(Create(self.axes), FadeIn(xl), FadeIn(yl), run_time=1.6)
        self.labels = VGroup(xl, yl)

        counts, ts = self.run.counts, self.run.times()
        pts = VGroup(*[Dot(self.axes.c2p(t, c), color=PART_C, radius=0.05)
                       for t, c in zip(ts, counts)])
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in pts],
                              lag_ratio=0.03, run_time=2.4))

        k = self.run.k
        curve = self.axes.plot(lambda t: N0 * np.exp(-k * t),
                               x_range=[0, STEPS, 0.5], color=CURVE_C,
                               stroke_width=4)
        eq = MathTex(rf"C(t) = {N0}\,e^{{-{k:.3f}\,t}}", font_size=32,
                     color=CURVE_C).to_corner(UR, buff=0.7)
        self.play(Create(curve), run_time=1.8)
        self.play(Write(eq), run_time=1.0)
        self.pts, self.curve, self.eq = pts, curve, eq
        self.wait(1.2)

    @beat("The half-life is the same wherever you start", seconds=20,
          narration="One number describes the whole curve. The half-life: the "
                    "time for half of whatever is present to go. From a "
                    "hundred and fifty down to seventy-five takes the same "
                    "time as seventy-five down to thirty-seven — and the same "
                    "again after that. It is log two over k, and it never "
                    "changes.")
    def half_life(self):
        h = self.run.half_life
        k = self.run.k

        marks = VGroup()
        level = float(N0)
        t = 0.0
        for _ in range(3):
            nxt_t, nxt_level = t + h, level / 2
            if nxt_t > STEPS:
                break
            marks.add(
                DashedLine(self.axes.c2p(t, level), self.axes.c2p(nxt_t, level),
                           color=DIM, stroke_width=2, dash_length=0.08),
                DashedLine(self.axes.c2p(nxt_t, level),
                           self.axes.c2p(nxt_t, nxt_level),
                           color=GONE_C, stroke_width=2.4, dash_length=0.08),
            )
            t, level = nxt_t, nxt_level

        self.play(FadeOut(self.eq), run_time=0.4)
        for i in range(0, len(marks), 2):
            self.play(Create(marks[i]), run_time=0.55)
            self.play(Create(marks[i + 1]), run_time=0.45)

        panel = VGroup(
            VGroup(MathTex(r"t_{1/2} = \frac{\ln 2}{k}", font_size=34,
                           color=GONE_C)),
            VGroup(MathTex(rf"= {h:.1f}", font_size=32, color=GONE_C),
                   Text("steps", font_size=19, color=DIM)
                   ).arrange(RIGHT, buff=0.22, aligned_edge=DOWN),
        ).arrange(DOWN, buff=0.28, aligned_edge=LEFT).to_corner(UR, buff=0.65)
        self.play(FadeIn(panel[0]), run_time=0.9)
        self.play(FadeIn(panel[1], shift=UP * 0.12), run_time=0.7)
        self.wait(2.2)
