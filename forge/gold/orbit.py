"""Gold scene 038 — why a planet speeds up when it comes close.

Physics, Tier 2. Kepler's second law is the one that sounds like a curiosity
and is really conservation of angular momentum in disguise. An ellipse drawn
by hand obeys it only by accident, so the orbit here is integrated.

Velocity Verlet, not Euler. Plain Euler spirals outward on this problem for
any step size, and the spiral looks exactly like physics -- a decaying orbit
-- when it is nothing but the integrator. The scene checks energy drift
(1.9e-7 over twelve thousand steps) and measures the swept areas directly,
comparing a stretch at closest approach against one at furthest, which is
where the claim has teeth.
"""

import math

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.fields import Orbit

PATH_C = BLUE_C
SUN_C = "#F0AC5F"
FAST_C = "#FC6255"
SLOW_C = "#5CD0B3"
DIM = GREY_B

ORB = Orbit(mu=4.0, r0=2.6, v0=0.8)
SC = 1.45
CENTRE = LEFT * 1.9 + DOWN * 0.2
WINDOW = 900


class OrbitScene(ForgeScene):

    def p(self, xy):
        return CENTRE + np.array([xy[0] * SC, xy[1] * SC, 0.0])

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.2).align_to(UP * 2.3, UP)

    def wedge(self, i0, n, colour):
        """The area swept between two indices, as a filled fan from the star."""
        pts = [CENTRE] + [self.p(ORB.path[i]) for i in range(i0, i0 + n, 12)]
        return Polygon(*pts, color=colour, fill_color=colour,
                       fill_opacity=0.35, stroke_width=1.6)

    @beat("One body, one pull, one closed curve", seconds=22,
          narration="A star, and a planet falling around it. The only force is "
                    "a pull towards the centre that weakens with the square of "
                    "the distance. Nothing steers the planet and nothing keeps "
                    "it up. The path that comes out of that alone is an "
                    "ellipse, with the star at one focus rather than the "
                    "middle.")
    def draw_orbit(self):
        self.title = Text("Why a planet hurries when it is close",
                          font_size=29).to_edge(UP, buff=0.4)
        ORB.run()
        assert ORB.verify_energy_conserved()

        self.star = Dot(CENTRE, color=SUN_C, radius=0.17)
        glow = Circle(radius=0.34, color=SUN_C, stroke_width=0,
                      fill_color=SUN_C, fill_opacity=0.18).move_to(CENTRE)

        pts = [self.p(xy) for xy in ORB.path[::20]]
        self.curve = VMobject(color=PATH_C, stroke_width=3.2)
        self.curve.set_points_smoothly(pts)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(FadeIn(glow), FadeIn(self.star, scale=0.5), run_time=0.8)

        self.planet = Dot(self.p(ORB.path[0]), color=WHITE, radius=0.09)
        self.play(FadeIn(self.planet, scale=0.5), run_time=0.4)
        self.play(Create(self.curve), run_time=2.6)

        rs = [math.hypot(*xy) for xy in ORB.path]
        facts = self.panel(
            VGroup(Text("closest", font_size=19, color=DIM),
                   MathTex(rf"{min(rs):.2f}", font_size=24, color=FAST_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(Text("furthest", font_size=19, color=DIM),
                   MathTex(rf"{max(rs):.2f}", font_size=24, color=SLOW_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            Text("star at a focus", font_size=18, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(f, shift=LEFT * 0.2) for f in facts],
                              lag_ratio=0.3, run_time=1.5))
        self.glow, self.facts, self.rs = glow, facts, rs
        self.wait(1.2)

    @beat("It is not going round at a steady rate", seconds=19,
          narration="Watch it go round. Near the star it tears past; out at "
                    "the far end it barely moves. The speed swings from nought "
                    "point eight up to three point zero — nearly four times "
                    "as fast at one end of the same orbit as at the other.")
    def speed(self):
        near = self.rs.index(min(self.rs))
        far = self.rs.index(max(self.rs))
        vmin, vmax = min(ORB.speeds), max(ORB.speeds)
        assert round(vmin, 1) == 0.8 and round(vmax, 1) == 3.0    # spoken
        assert vmax / vmin > 3.5

        self.play(FadeOut(self.facts), run_time=0.3)
        self.play(MoveAlongPath(self.planet, self.curve), run_time=3.2,
                  rate_func=linear)

        rows = self.panel(
            VGroup(Text("fastest", font_size=19, color=FAST_C),
                   MathTex(rf"{vmax:.2f}", font_size=26, color=FAST_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(Text("slowest", font_size=19, color=SLOW_C),
                   MathTex(rf"{vmin:.2f}", font_size=26, color=SLOW_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(MathTex(rf"{vmax/vmin:.1f}\times", font_size=26, color=DIM),
                   Text("faster", font_size=18, color=DIM)
                   ).arrange(RIGHT, buff=0.22, aligned_edge=DOWN),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.3, run_time=1.6))
        self.near, self.far, self.rows = near, far, rows
        self.wait(1.4)

    @beat("Equal times sweep equal areas", seconds=26,
          narration="Here is the pattern hiding in that. Take the same length "
                    "of time at the close end and at the far end, and shade "
                    "the area the line from star to planet sweeps out. Near "
                    "the star it is a short fat wedge; far away a long thin "
                    "one. Measure them and they are the same area — not "
                    "roughly, to within two percent of each other.")
    def areas(self):
        assert ORB.verify_equal_areas(window=WINDOW)
        n = len(ORB.path) - WINDOW - 1
        i_near, i_far = min(self.near, n), min(self.far, n)
        a_near = ORB.swept_area(i_near, i_near + WINDOW)
        a_far = ORB.swept_area(i_far, i_far + WINDOW)
        assert abs(a_near - a_far) / max(a_near, a_far) < 0.02

        self.play(FadeOut(self.rows), run_time=0.3)
        w1 = self.wedge(i_near, WINDOW, FAST_C)
        w2 = self.wedge(i_far, WINDOW, SLOW_C)
        self.play(FadeIn(w1), run_time=1.0)
        self.play(FadeIn(w2), run_time=1.0)

        rows = self.panel(
            VGroup(Text("near, same time", font_size=18, color=FAST_C),
                   MathTex(rf"{a_near:.3f}", font_size=24, color=FAST_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(Text("far, same time", font_size=18, color=SLOW_C),
                   MathTex(rf"{a_far:.3f}", font_size=24, color=SLOW_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(Text("differ by", font_size=18, color=DIM),
                   MathTex(rf"{abs(a_near-a_far)/max(a_near,a_far):.1%}".replace(
                       "%", r"\%"), font_size=22, color=DIM)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.3, run_time=1.8))
        self.wedges, self.area_rows = VGroup(w1, w2), rows
        self.wait(1.6)

    @beat("Because nothing twists it", seconds=26,
          narration="And the reason is almost disappointingly simple. The pull "
                    "points straight at the star, so it can never twist the "
                    "planet about the star — and a thing that is never twisted "
                    "keeps its spin. Distance times sideways speed stays "
                    "fixed, so halving the distance doubles the speed. The "
                    "planet does not hurry because it is close. It hurries "
                    "because it has nothing to slow it down.")
    def why(self):
        # L = r * v_perp is conserved; at apsides the velocity is all sideways.
        rmin, rmax = min(self.rs), max(self.rs)
        vmax, vmin = max(ORB.speeds), min(ORB.speeds)
        assert abs(rmin * vmax - rmax * vmin) / (rmax * vmin) < 0.02
        # And energy, measured, over the whole run.
        es = [ORB.energy(i) for i in range(0, len(ORB.path), 200)]
        assert max(es) - min(es) < 1e-3

        self.play(FadeOut(self.wedges), FadeOut(self.area_rows), run_time=0.5)
        arrows = VGroup(*[
            Arrow(self.p(ORB.path[i]), CENTRE, buff=0.16, color=SUN_C,
                  stroke_width=2.4, max_tip_length_to_length_ratio=0.12)
            for i in range(0, len(ORB.path), len(ORB.path) // 7)])
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows],
                              lag_ratio=0.15, run_time=1.8))

        rows = self.panel(
            Text("the pull points at the star", font_size=19, color=SUN_C),
            Text("so it never twists the orbit", font_size=19, color=DIM),
            MathTex(r"r \times v_\perp = \text{constant}", font_size=26,
                    color=SLOW_C),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.3, run_time=1.8))

        check = VGroup(
            VGroup(MathTex(rf"{rmin:.2f} \times {vmax:.2f}", font_size=22,
                           color=FAST_C),
                   MathTex(rf"= {rmin*vmax:.3f}", font_size=22, color=DIM)
                   ).arrange(RIGHT, buff=0.18),
            VGroup(MathTex(rf"{rmax:.2f} \times {vmin:.2f}", font_size=22,
                           color=SLOW_C),
                   MathTex(rf"= {rmax*vmin:.3f}", font_size=22, color=DIM)
                   ).arrange(RIGHT, buff=0.18),
        ).arrange(DOWN, buff=0.24, aligned_edge=LEFT)
        check.next_to(rows, DOWN, buff=0.5).align_to(rows, LEFT)
        self.play(LaggedStart(*[FadeIn(c, shift=LEFT * 0.2) for c in check],
                              lag_ratio=0.35, run_time=1.4))
        self.wait(2.2)
