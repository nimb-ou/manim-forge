"""Gold scene 040 — waves add, and that is the whole of interference.

Physics, Tier 2. Interference is often introduced with photographs of ripple
tanks, which show the result and hide the rule. The rule is one line: at every
point, add the displacements. Everything else -- cancellation, reinforcement,
standing waves, nodes -- is a consequence of that single sum.

So the combined wave here is computed by adding samples, never by drawing an
envelope. ``verify_superposition_is_linear`` asserts the sum-of-waves equals
the wave-of-sums at sixty points, which sounds trivial and is exactly what an
animation gets wrong when someone sketches the envelope by hand and the two
quietly disagree.

The node positions come from solving sin(kx) = 0 rather than from reading the
picture; a node drawn slightly off is invisible and wrong.
"""

import math

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.fields import (standing_nodes, superpose,
                                     verify_superposition_is_linear, wave)

A_C = BLUE_C
B_C = "#F0AC5F"
SUM_C = "#5CD0B3"
NODE_C = "#FC6255"
DIM = GREY_B

K = 2.0
OMEGA = 2.0
SPAN = 5.6
RIGHTWARD = dict(amp=1.0, k=K, omega=OMEGA, direction=1)
LEFTWARD = dict(amp=1.0, k=K, omega=OMEGA, direction=-1)


class WaveInterference(ForgeScene):

    def make_axes(self, y_top, height=1.35):
        return Axes(x_range=[-SPAN, SPAN, 2], y_range=[-2.4, 2.4, 2],
                    x_length=7.4, y_length=height,
                    axis_config={"include_tip": False, "stroke_color": GREY_E,
                                 "stroke_width": 1.4, "font_size": 14},
                    ).move_to(LEFT * 1.4 + UP * y_top)

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.6).align_to(UP * 2.3, UP)

    @beat("Two waves, going opposite ways", seconds=16,
          narration="Two waves of the same size and the same wavelength, one "
                    "travelling right and one travelling left. Separately "
                    "there is nothing remarkable about either. The question is "
                    "what happens where they overlap, and the answer is "
                    "simpler than it looks.")
    def two_waves(self):
        self.title = Text("Waves add", font_size=30).to_edge(UP, buff=0.4)
        self.t = ValueTracker(0.0)

        self.ax_a = self.make_axes(1.75)
        self.ax_b = self.make_axes(0.35)
        self.ax_s = self.make_axes(-1.5, height=1.9)

        self.wa = always_redraw(lambda: self.ax_a.plot(
            lambda x: wave(x, self.t.get_value(), **RIGHTWARD),
            x_range=[-SPAN, SPAN, 0.04], color=A_C, stroke_width=3.4))
        self.wb = always_redraw(lambda: self.ax_b.plot(
            lambda x: wave(x, self.t.get_value(), **LEFTWARD),
            x_range=[-SPAN, SPAN, 0.04], color=B_C, stroke_width=3.4))

        labs = VGroup(
            Text("travelling right", font_size=17, color=A_C).next_to(
                self.ax_a, LEFT, buff=0.25),
            Text("travelling left", font_size=17, color=B_C).next_to(
                self.ax_b, LEFT, buff=0.25))

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.ax_a), Create(self.ax_b), run_time=1.0)
        self.play(Create(self.wa), Create(self.wb), run_time=1.6)
        self.play(FadeIn(labs), run_time=0.6)
        self.play(self.t.animate.set_value(1.6), run_time=2.4, rate_func=linear)
        self.labs = labs
        self.wait(0.8)

    @beat("At every point, add the two heights", seconds=23,
          narration="Here is the rule, and it is all of it. Pick any point "
                    "along the line. Read the height of the first wave there, "
                    "read the height of the second, add them. That sum is the "
                    "height of the combined wave at that point. Do it "
                    "everywhere and you have the answer — no new physics, just "
                    "addition.")
    def add_them(self):
        assert verify_superposition_is_linear([RIGHTWARD, LEFTWARD])

        self.play(Create(self.ax_s), run_time=0.8)
        self.ws = always_redraw(lambda: self.ax_s.plot(
            lambda x: superpose(x, self.t.get_value(), [RIGHTWARD, LEFTWARD]),
            x_range=[-SPAN, SPAN, 0.04], color=SUM_C, stroke_width=4))
        self.play(Create(self.ws), run_time=1.4)

        probe_x = 1.15
        def marker(ax, parts, colour):
            return always_redraw(lambda: Dot(
                ax.c2p(probe_x, sum(wave(probe_x, self.t.get_value(), **p)
                                    for p in parts)),
                color=colour, radius=0.075))
        m_a = marker(self.ax_a, [RIGHTWARD], A_C)
        m_b = marker(self.ax_b, [LEFTWARD], B_C)
        m_s = marker(self.ax_s, [RIGHTWARD, LEFTWARD], SUM_C)
        self.play(FadeIn(m_a, scale=0.5), FadeIn(m_b, scale=0.5),
                  FadeIn(m_s, scale=0.5), run_time=0.8)

        readout = self.panel(
            always_redraw(lambda: VGroup(
                MathTex(rf"{wave(probe_x, self.t.get_value(), **RIGHTWARD):+.2f}",
                        font_size=24, color=A_C),
                MathTex(rf"{wave(probe_x, self.t.get_value(), **LEFTWARD):+.2f}",
                        font_size=24, color=B_C),
                MathTex(rf"= {superpose(probe_x, self.t.get_value(), [RIGHTWARD, LEFTWARD]):+.2f}",
                        font_size=26, color=SUM_C),
            ).arrange(DOWN, buff=0.22, aligned_edge=RIGHT).move_to(
                RIGHT * 4.6 + UP * 1.7)))
        self.play(FadeIn(readout), run_time=0.8)
        self.play(self.t.animate.set_value(3.4), run_time=3.0, rate_func=linear)
        self.markers = VGroup(m_a, m_b, m_s)
        self.readout = readout
        self.wait(0.8)

    @beat("Some places never move at all", seconds=19,
          narration="Watch the combined wave rather than the parts. It is no "
                    "longer travelling anywhere — it stands still and breathes "
                    "in place. And certain points never move at all, at any "
                    "moment. Those are the nodes, and they sit exactly where "
                    "the two waves are always equal and opposite.")
    def nodes(self):
        ns = [n for n in standing_nodes(K, SPAN) if abs(n) <= SPAN - 0.2]
        # Every node must genuinely stay at zero for all t, not merely look still.
        for n in ns:
            for i in range(12):
                assert abs(superpose(n, i * 0.37, [RIGHTWARD, LEFTWARD])) < 1e-9

        self.play(FadeOut(self.markers), FadeOut(self.readout),
                  FadeOut(self.labs), run_time=0.5)
        self.play(FadeOut(self.ax_a), FadeOut(self.ax_b),
                  FadeOut(self.wa), FadeOut(self.wb), run_time=0.7)
        self.play(self.ax_s.animate.move_to(LEFT * 1.4 + DOWN * 0.3).scale(1.15),
                  run_time=1.0)

        marks = VGroup(*[
            Dot(self.ax_s.c2p(n, 0), color=NODE_C, radius=0.08) for n in ns])
        rails = VGroup(*[
            DashedLine(self.ax_s.c2p(n, -2.2), self.ax_s.c2p(n, 2.2),
                       color=NODE_C, stroke_width=1.4, dash_length=0.08,
                       stroke_opacity=0.55) for n in ns])
        self.play(LaggedStart(*[Create(r) for r in rails],
                              lag_ratio=0.1, run_time=1.4))
        self.play(LaggedStart(*[FadeIn(m, scale=0.5) for m in marks],
                              lag_ratio=0.1, run_time=1.0))
        self.play(self.t.animate.set_value(6.6), run_time=3.2, rate_func=linear)

        note = self.panel(
            VGroup(MathTex(rf"{len(ns)}", font_size=28, color=NODE_C),
                   Text("nodes", font_size=20, color=DIM)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            Text("never move, ever", font_size=19, color=NODE_C),
            Text("fixed in space", font_size=18, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(n, shift=LEFT * 0.2) for n in note],
                              lag_ratio=0.3, run_time=1.4))
        self.node_group = VGroup(marks, rails)
        self.note, self.ns = note, ns
        self.wait(1.2)

    @beat("Where the nodes are, and why", seconds=28,
          narration="Their positions are not a matter of looking carefully. "
                    "Adding the two waves algebraically gives twice sine k x, "
                    "times cosine omega t — and that splits cleanly into a "
                    "part that depends only on position and a part that "
                    "depends only on time. The wave is nailed down in space "
                    "and free to breathe in time. It vanishes wherever sine k "
                    "x is zero, which is every half wavelength, exactly.")
    def why(self):
        half_lambda = math.pi / K
        gaps = [round(b - a, 9) for a, b in zip(self.ns, self.ns[1:])]
        assert len(set(gaps)) == 1 and abs(gaps[0] - half_lambda) < 1e-9

        self.play(FadeOut(self.note), run_time=0.3)
        ident = VGroup(
            MathTex(r"\sin(kx - \omega t) + \sin(kx + \omega t)",
                    font_size=26, color=DIM),
            MathTex(r"= 2\,\sin(kx)\,\cos(\omega t)", font_size=30, color=SUM_C),
        ).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        ident.move_to(RIGHT * 4.3 + UP * 1.8)
        self.play(Write(ident[0]), run_time=1.3)
        self.play(FadeIn(ident[1], shift=UP * 0.12), run_time=1.0)

        split = VGroup(
            VGroup(MathTex(r"\sin(kx)", font_size=24, color=NODE_C),
                   Text("position only", font_size=17, color=DIM)
                   ).arrange(RIGHT, buff=0.25),
            VGroup(MathTex(r"\cos(\omega t)", font_size=24, color=SUM_C),
                   Text("time only", font_size=17, color=DIM)
                   ).arrange(RIGHT, buff=0.25),
        ).arrange(DOWN, buff=0.24, aligned_edge=LEFT)
        split.next_to(ident, DOWN, buff=0.5).align_to(ident, LEFT)
        self.play(LaggedStart(*[FadeIn(s, shift=LEFT * 0.2) for s in split],
                              lag_ratio=0.35, run_time=1.4))

        spacing = VGroup(
            VGroup(Text("node spacing", font_size=18, color=DIM),
                   MathTex(rf"{half_lambda:.3f}", font_size=24, color=NODE_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            MathTex(r"= \tfrac{1}{2}\lambda", font_size=26, color=NODE_C),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        spacing.next_to(split, DOWN, buff=0.5).align_to(split, LEFT)
        self.play(FadeIn(spacing, shift=UP * 0.12), run_time=1.0)
        self.play(self.t.animate.set_value(9.8), run_time=2.8, rate_func=linear)
        self.wait(1.6)
