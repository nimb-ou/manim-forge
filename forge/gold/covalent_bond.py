"""Gold scene 015 — a covalent bond as the bottom of an energy curve.

Chemistry, Tier 2. The usual picture of a bond -- two circles overlapping with
a pair of dots in the middle -- says what a bond looks like and nothing about
why it exists. This scene puts the energy on an axis, because "the bond is the
minimum" is the only version of the idea that explains both why atoms stick
together and why a bond has a particular length.

The curve is a Morse potential rather than a parabola. A parabola would make
pulling the atoms apart cost unbounded energy, so the scene's closing point --
that 436 kilojoules per mole breaks the bond -- would be a claim its own graph
contradicts.

The minimum is found by scanning the curve, not by drawing a dot at r_e and
trusting the parameters. Bond length and dissociation energy are measured
values, flagged as measured in the primitive; everything else is derived.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.chem import (H2_BOND_LENGTH_PM, H2_DISSOCIATION_KJ,
                                   MorsePotential)

ATOM_C = "#FC6255"
E_C = BLUE_C
CURVE_C = "#5CD0B3"
HI_C = "#F0AC5F"
DIM = GREY_B

R_MIN, R_MAX = 45.0, 260.0
ATOM_Y = 2.55                 # where the two atoms ride, above the axes


class CovalentBond(ForgeScene):

    pot = MorsePotential()

    def atom_x(self, r):
        """Screen x for an atom at half-separation ``r``/2, centred on 0."""
        return (r / H2_BOND_LENGTH_PM) * 0.62

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.0).align_to(UP * 1.4, UP)

    @beat("Two atoms, far apart, ignoring each other", seconds=15,
          narration="Two hydrogen atoms, a long way apart. Neither one feels "
                    "the other, so there is nothing to gain by moving and "
                    "nothing to lose. Call that energy zero — it is the "
                    "reference everything else is measured against.")
    def apart(self):
        self.title = Text("What holds two atoms together?",
                          font_size=30).to_edge(UP, buff=0.4)

        self.axes = Axes(
            x_range=[R_MIN, R_MAX, 50], y_range=[-520, 260, 200],
            x_length=7.4, y_length=3.9,
            axis_config={"include_tip": False, "stroke_color": GREY_B,
                         "stroke_width": 2, "font_size": 20},
            y_axis_config={"numbers_to_include": [-400, -200, 0, 200]},
            x_axis_config={"numbers_to_include": [50, 100, 150, 200, 250]},
        ).shift(DOWN * 1.25 + LEFT * 1.1)
        self.x_lab = self.axes.get_x_axis_label(
            Text("separation (pm)", font_size=19, color=DIM), edge=DOWN,
            direction=DOWN, buff=0.3)
        self.y_lab = self.axes.get_y_axis_label(
            Text("energy (kJ/mol)", font_size=19, color=DIM), edge=UP,
            direction=UP, buff=0.2)

        self.r = ValueTracker(240.0)
        self.left = always_redraw(lambda: Circle(
            radius=0.3, color=ATOM_C, fill_opacity=0.35, stroke_width=2.5
        ).move_to([-self.atom_x(self.r.get_value()), ATOM_Y, 0]))
        self.right = always_redraw(lambda: Circle(
            radius=0.3, color=ATOM_C, fill_opacity=0.35, stroke_width=2.5
        ).move_to([self.atom_x(self.r.get_value()), ATOM_Y, 0]))

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.axes), FadeIn(self.x_lab), FadeIn(self.y_lab),
                  run_time=1.6)
        self.play(FadeIn(self.left, scale=0.5), FadeIn(self.right, scale=0.5),
                  run_time=0.8)

        zero = DashedLine(self.axes.c2p(R_MIN, 0), self.axes.c2p(R_MAX, 0),
                          color=GREY_C, stroke_width=1.6, dash_length=0.1)
        self.play(Create(zero), run_time=0.9)
        self.zero = zero
        self.wait(1.2)

    @beat("Bringing them closer releases energy", seconds=17,
          narration="Now push them together. Each electron starts to feel "
                    "both nuclei at once, and being pulled from two sides is a "
                    "lower-energy place to be than being pulled from one. The "
                    "curve falls. The atoms are not being forced together — "
                    "they are falling.")
    def approach(self):
        self.curve = self.axes.plot(lambda r: self.pot.energy(r),
                                    x_range=[R_MIN, R_MAX, 1.5],
                                    color=CURVE_C, stroke_width=4)
        self.marker = always_redraw(lambda: Dot(
            self.axes.c2p(self.r.get_value(),
                          self.pot.energy(self.r.get_value())),
            color=HI_C, radius=0.085))

        self.play(Create(self.curve), run_time=2.2)
        self.play(FadeIn(self.marker, scale=0.5), run_time=0.5)
        self.play(self.r.animate.set_value(H2_BOND_LENGTH_PM),
                  run_time=3.4, rate_func=smooth)
        self.wait(1.4)

    @beat("Pushing further costs more than it gains", seconds=18,
          narration="Keep pushing and the curve turns around. The two nuclei "
                    "are both positive, and at close range that repulsion beats "
                    "everything else. So there is a distance that is better "
                    "than being apart and better than being closer — and that "
                    "is what a bond is.")
    def repulsion(self):
        self.play(self.r.animate.set_value(52.0), run_time=1.8,
                  rate_func=smooth)
        self.wait(0.8)
        self.play(self.r.animate.set_value(H2_BOND_LENGTH_PM), run_time=1.6,
                  rate_func=smooth)

        # Scanned, not assumed: if the parameters ever drift, the dot would
        # land somewhere the curve does not actually bottom out.
        r_min, v_min = self.pot.minimum()
        assert self.pot.verify_minimum_at_re()

        drop = DashedLine(self.axes.c2p(r_min, 0), self.axes.c2p(r_min, v_min),
                          color=HI_C, stroke_width=2, dash_length=0.09)
        self.play(Create(drop), run_time=0.9)
        self.drop = drop
        self.wait(1.2)

    @beat("The bottom of the curve is the bond", seconds=18,
          narration="The lowest point sits at seventy-four picometres apart, "
                    "four hundred and thirty-six kilojoules per mole below "
                    "where the atoms started. That depth is the bond: supply "
                    "that much energy and the two atoms climb back out and go "
                    "their separate ways. Supply less, and they stay.")
    def minimum(self):
        r_min, v_min = self.pot.minimum()

        panel = self.panel(
            VGroup(Text("bond length", font_size=20, color=DIM),
                   MathTex(rf"{r_min:.0f}\,\text{{pm}}", font_size=30, color=HI_C)
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
            VGroup(Text("depth", font_size=20, color=DIM),
                   MathTex(rf"{abs(v_min):.0f}", font_size=30, color=CURVE_C)
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
            Text("kJ/mol to break it", font_size=18, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(p, shift=LEFT * 0.2) for p in panel],
                              lag_ratio=0.35, run_time=1.6))

        depth = DoubleArrow(self.axes.c2p(r_min, 0),
                            self.axes.c2p(r_min, v_min),
                            color=CURVE_C, stroke_width=3,
                            tip_length=0.16, buff=0)
        self.play(FadeOut(self.drop), Create(depth), run_time=1.2)

        # Break it: climb back out to the flat region the curve really has.
        self.play(self.r.animate.set_value(250.0), run_time=2.6,
                  rate_func=smooth)
        broke = Text("bond broken", font_size=22, color=ATOM_C).next_to(
            panel, DOWN, buff=0.55).align_to(panel, LEFT)
        self.play(FadeIn(broke, shift=UP * 0.15), run_time=0.8)
        self.wait(1.8)
