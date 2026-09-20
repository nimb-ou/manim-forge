"""Gold scene 023 — unrolling a circle into a rectangle.

Geometry, Tier 2. Cut a disc into sectors, interleave them points-up and
points-down, and the result approaches a rectangle of width pi r and height r.
Its area is pi r squared, which is where the formula comes from.

The honest part is the height. Each sector's straight height is r cos(pi/n),
not r -- it only becomes r in the limit -- so for eight sectors the shape is
visibly not a rectangle and its area is visibly short of pi r squared. Most
tellings quietly label the height r from the first picture, which turns a
limiting argument into a false one. Here the shortfall is computed and shown
shrinking, so the limit is something the viewer watches happen.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.geometry import (SectorRearrangement, sector_convergence,
                                       verify_sectors_converge)

DISC_C = BLUE_C
ALT_C = "#F0AC5F"
EXACT_C = "#5CD0B3"
ERR_C = "#FC6255"
DIM = GREY_B

R = 1.7                       # screen radius of the disc
STAGES = [8, 16, 32]


class CircleArea(ForgeScene):

    def sectors(self, n, centre, colour_split=True):
        """``n`` wedges of the disc, as a VGroup, in drawing order."""
        out = VGroup()
        for i in range(n):
            start = TAU * i / n
            w = AnnularSector(inner_radius=0, outer_radius=R,
                              angle=TAU / n, start_angle=start,
                              color=ALT_C if (colour_split and i % 2) else DISC_C,
                              fill_opacity=0.7, stroke_width=1.2,
                              stroke_color=BLACK).move_arc_center_to(centre)
            out.add(w)
        return out

    def laid_out(self, n, left_edge, y):
        """Where wedge ``i`` goes once interleaved into the strip."""
        geo = SectorRearrangement(n, R)
        step = geo.width * 2 / n          # each pair contributes one full arc
        spots = []
        for i in range(n):
            x = left_edge + step * (i // 2) + (step / 2 if i % 2 else 0)
            spots.append(np.array([x, y + (0.0 if i % 2 == 0 else geo.height),
                                   0.0]))
        return spots, geo

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.4).align_to(UP * 2.3, UP)

    @beat("A disc, and the number we want", seconds=19,
          narration="A disc of radius r. Everyone knows its area is pi r "
                    "squared, and almost nobody is shown why. The argument is "
                    "a rearrangement: cut the disc into wedges and lay them "
                    "out differently, without losing or gaining any area, "
                    "until the shape is one whose area is obvious.")
    def disc(self):
        self.title = Text("Why the area of a circle is pi r squared",
                          font_size=28).to_edge(UP, buff=0.4)
        centre = LEFT * 3.4 + DOWN * 0.3

        self.circle = Circle(radius=R, color=DISC_C, fill_color=DISC_C,
                             fill_opacity=0.45, stroke_width=3).move_to(centre)
        radius = Line(centre, centre + RIGHT * R, color=WHITE, stroke_width=2.5)
        rlab = MathTex("r", font_size=26, color=WHITE).next_to(radius, UP,
                                                               buff=0.1)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(GrowFromCenter(self.circle), run_time=1.2)
        self.play(Create(radius), Write(rlab), run_time=0.9)

        goal = self.panel(
            MathTex(r"A = \pi r^2", font_size=36, color=EXACT_C),
            Text("to be shown, not assumed", font_size=18, color=DIM),
        )
        self.play(Write(goal[0]), run_time=1.0)
        self.play(FadeIn(goal[1]), run_time=0.6)
        self.centre, self.goal = centre, goal
        self.radius_marks = VGroup(radius, rlab)
        self.wait(1.4)

    @beat("Cut it into wedges and interleave them", seconds=20,
          narration="Cut the disc into eight equal wedges and lay them out in "
                    "a row, alternating point up and point down so they "
                    "interlock. Not one scrap of area has been added or "
                    "removed — the same eight pieces are simply somewhere "
                    "else. The shape they make is nearly a parallelogram.")
    def unroll(self):
        n = STAGES[0]
        self.play(FadeOut(self.circle), FadeOut(self.radius_marks),
                  run_time=0.5)

        wedges = self.sectors(n, self.centre)
        self.play(LaggedStart(*[FadeIn(w, scale=0.85) for w in wedges],
                              lag_ratio=0.08, run_time=1.6))

        spots, geo = self.laid_out(n, -4.6, -1.5)
        anims = []
        for i, (w, spot) in enumerate(zip(wedges, spots)):
            target = w.copy()
            if i % 2:
                target.rotate(PI, about_point=target.get_center())
            target.move_to(spot)
            anims.append(Transform(w, target))
        self.play(LaggedStart(*anims, lag_ratio=0.07, run_time=2.8))
        self.wedges, self.n_now = wedges, n
        self.wait(1.2)

    @beat("The shape is short, and by how much", seconds=27,
          narration="Here is the part usually skipped. The width really is "
                    "pi r — half the circumference, laid end to end. But the "
                    "height is not r. It is r times the cosine of pi over n, "
                    "because each wedge is a triangle with a curved top, and "
                    "for eight wedges that is about nine tenths of r. The "
                    "rearranged shape falls short of pi r squared, and it "
                    "should.")
    def shortfall(self):
        geo = SectorRearrangement(self.n_now, R)
        unit = geo.exact / (PI * R ** 2)          # == 1; areas are in screen units

        rows = VGroup(
            VGroup(Text("width", font_size=20, color=DIM),
                   MathTex(r"\pi r", font_size=28, color=EXACT_C)
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
            VGroup(Text("height", font_size=20, color=DIM),
                   MathTex(rf"r\cos\frac{{\pi}}{{{self.n_now}}}", font_size=28,
                           color=ERR_C)
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
            VGroup(MathTex(rf"= {geo.height / R:.3f}\,r", font_size=24,
                           color=ERR_C)),
        ).arrange(DOWN, buff=0.28, aligned_edge=LEFT)

        gap = VGroup(
            Text("short by", font_size=20, color=DIM),
            MathTex(rf"{geo.error / geo.exact:.1%}".replace("%", r"\%"),
                    font_size=30, color=ERR_C),
        ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN)

        self.play(FadeOut(self.goal), run_time=0.3)
        panel = self.panel(rows, gap)
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.3, run_time=1.8))
        self.play(FadeIn(gap, shift=UP * 0.12), run_time=0.8)
        self.shortfall_panel = panel
        self.wait(1.8)

    @beat("Thinner wedges, and the shortfall vanishes", seconds=25,
          narration="Now use more wedges. Sixteen, then thirty-two. The "
                    "zig-zag along the top flattens, the height climbs towards "
                    "r, and the shortfall falls away — under two percent, then "
                    "under half a percent. In the limit the shape is exactly a "
                    "rectangle, pi r wide and r high, and its area is pi r "
                    "squared. That is the formula, and that is where it comes "
                    "from.")
    def converge(self):
        assert verify_sectors_converge(R)

        rows = VGroup()
        for n, area, err in sector_convergence(STAGES + [512], R):
            rows.add(VGroup(
                MathTex(rf"n = {n}", font_size=22, color=DIM),
                MathTex(rf"{err / (PI * R ** 2):.2%}".replace("%", r"\%"), font_size=24,
                        color=ERR_C if n < 100 else EXACT_C),
                Text("short", font_size=17, color=DIM),
            ).arrange(RIGHT, buff=0.22))
        rows.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        rows.move_to(RIGHT * 4.4 + UP * 0.3)

        self.play(FadeOut(self.shortfall_panel), run_time=0.4)
        self.play(FadeIn(rows[0], shift=LEFT * 0.2), run_time=0.5)

        for i, n in enumerate(STAGES[1:], start=1):
            new = self.sectors(n, self.centre)
            spots, geo = self.laid_out(n, -4.6, -1.5)
            for j, (w, spot) in enumerate(zip(new, spots)):
                if j % 2:
                    w.rotate(PI, about_point=w.get_center())
                w.move_to(spot)
            self.play(FadeOut(self.wedges), FadeIn(new), run_time=0.9)
            self.wedges = new
            self.play(FadeIn(rows[i], shift=LEFT * 0.2), run_time=0.5)
            self.wait(0.5)

        self.play(FadeIn(rows[len(STAGES)], shift=LEFT * 0.2), run_time=0.6)
        answer = MathTex(r"A = \pi r \cdot r = \pi r^2", font_size=34,
                         color=EXACT_C).next_to(rows, DOWN, buff=0.6)
        self.play(Write(answer), run_time=1.4)
        self.wait(2.2)
