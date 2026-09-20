"""Gold scene 012 — why a triangle's angles make a straight line.

Tier 1. The tearing-corners argument is the first proof most people meet that
is genuinely convincing without algebra, and it establishes the habit every
later geometry scene relies on: move the pieces, see the result.

The angles are measured from the triangle's own vertices and asserted to sum
to pi before the scene claims they do — so an arbitrary triangle would still
be shown honestly.
"""

import math

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.vectors import Vec2

A_C = "#FC6255"
B_C = "#5CD0B3"
C_C = "#F0AC5F"
DIM = GREY_B


class AngleSum(ForgeScene):

    # A deliberately irregular triangle: a neat one invites the suspicion that
    # the result depends on the shape.
    pts = [Vec2(-3.0, -1.5), Vec2(2.8, -1.3), Vec2(1.4, 1.9)]

    def angles(self) -> list[float]:
        a, b, c = self.pts
        return [(b - a).angle_to(c - a),
                (a - b).angle_to(c - b),
                (a - c).angle_to(b - c)]

    @beat("Any triangle at all", seconds=10,
          narration="Any triangle. Not a special one — this has no equal sides "
                    "and no right angle. It has three corners, and each corner "
                    "has an angle.")
    def draw(self):
        a, b, c = self.pts
        P = [np.array([p.x, p.y, 0.0]) for p in self.pts]
        self.tri = Polygon(*P, color=BLUE_C, stroke_width=3.5,
                           fill_opacity=0.10, fill_color=BLUE_C)
        self.title = Text("Why do they always add to a straight line?",
                          font_size=28).to_edge(UP, buff=0.4)

        self.arcs = VGroup()
        for i, col in enumerate((A_C, B_C, C_C)):
            v = P[i]
            u1 = (P[(i + 1) % 3] - v); u1 /= np.linalg.norm(u1)
            u2 = (P[(i + 2) % 3] - v); u2 /= np.linalg.norm(u2)
            self.arcs.add(Angle(Line(v, v + u1), Line(v, v + u2),
                                radius=0.48, color=col, stroke_width=5))

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.tri), run_time=1.4)
        self.play(LaggedStart(*[Create(a) for a in self.arcs],
                              lag_ratio=0.25, run_time=1.5))
        self.wait(0.7)

    @beat("Measure them", seconds=13,
          narration="Measure each one and add them up. One hundred and eighty "
                    "degrees — a straight line. Try it with a different "
                    "triangle, any triangle at all, and you get the same "
                    "total.")
    def measure(self):
        angs = self.angles()
        # The scene's claim, checked against the triangle's own geometry.
        assert abs(sum(angs) - math.pi) < 1e-9

        rows = VGroup()
        for ang, col in zip(angs, (A_C, B_C, C_C)):
            rows.add(Text(f"{math.degrees(ang):.0f}", font_size=34, color=col))
        rows.arrange(RIGHT, buff=0.5).to_edge(RIGHT, buff=1.1).shift(UP * 1.4)

        plus = VGroup(*[MathTex("+", font_size=28, color=DIM) for _ in range(2)])
        for i, p in enumerate(plus):
            p.move_to((rows[i].get_center() + rows[i + 1].get_center()) / 2)

        total = VGroup(
            Text(f"{math.degrees(sum(angs)):.0f}", font_size=60, color=WHITE),
            Text("a straight line", font_size=21, color=DIM),
        ).arrange(DOWN, buff=0.14).next_to(rows, DOWN, buff=0.6)

        self.play(LaggedStart(*[FadeIn(r, scale=0.7) for r in rows],
                              lag_ratio=0.25, run_time=1.2))
        self.play(FadeIn(plus), run_time=0.4)
        self.play(FadeIn(total[0], scale=0.7), Write(total[1]), run_time=1.0)
        self.total = VGroup(rows, plus, total)
        self.wait(0.8)

    @beat("Tear the corners off and fit them together", seconds=20,
          narration="But why? Tear the three corners off and bring them "
                    "together at a point. They fit, with no gap and no overlap, "
                    "along a single straight edge. The angles were always going "
                    "to do this, because the two sides leaving any corner stay "
                    "parallel to the sides of the others.")
    def tear(self):
        P = [np.array([p.x, p.y, 0.0]) for p in self.pts]
        angs = self.angles()
        meet = np.array([-2.9, -2.7, 0.0])

        wedges = VGroup()
        running = 0.0
        for i, col in enumerate((A_C, B_C, C_C)):
            v = P[i]
            u1 = (P[(i + 1) % 3] - v); u1 /= np.linalg.norm(u1)
            start = math.atan2(u1[1], u1[0])
            wedges.add(AnnularSector(inner_radius=0, outer_radius=0.95,
                                     angle=angs[i], start_angle=start,
                                     color=col, fill_opacity=0.65, stroke_width=0
                                     ).move_arc_center_to(v))
            running += angs[i]

        self.play(FadeOut(self.arcs), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(w) for w in wedges], lag_ratio=0.2, run_time=1.2))

        # Lay them side by side around a common point, in angle order.
        laid = VGroup()
        acc = 0.0
        for i, w in enumerate(wedges):
            target = AnnularSector(inner_radius=0, outer_radius=0.95,
                                   angle=angs[i], start_angle=acc,
                                   color=w.get_color(), fill_opacity=0.65,
                                   stroke_width=0).move_arc_center_to(meet)
            laid.add(target)
            acc += angs[i]

        self.play(*[ReplacementTransform(w, t) for w, t in zip(wedges, laid)],
                  self.tri.animate.set_stroke(opacity=0.3).set_fill(opacity=0.04),
                  run_time=2.2)

        edge = Line(meet + LEFT * 1.25, meet + RIGHT * 1.25,
                    color=WHITE, stroke_width=3.5)
        self.play(Create(edge), run_time=0.9)
        cap = Text("no gap, no overlap", font_size=21, color=DIM)
        cap.next_to(edge, DOWN, buff=0.28)
        self.play(FadeIn(cap, shift=UP * 0.12), run_time=0.8)
        self.wait(1.8)
