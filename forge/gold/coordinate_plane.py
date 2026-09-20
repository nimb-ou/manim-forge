"""Gold scene 013 — two numbers locate a point.

Tier 1, and the last of the foundations. Every scene that graphs anything
assumes a plane has been drawn well: axes that read as axes rather than as
vectors, a labelled origin, guide lines that show where a coordinate came
from. A model that has never seen this done carefully will draw it carelessly
inside a scene about something else, and the error will be invisible because
the scene will be *about* the derivative, not about the axes.

Every number spoken or shown is computed. The coordinates are read off the
positions actually drawn; the distance comes from ``plane.distance`` and is
checked against Pythagoras before the scene asserts it; the twelve lattice
points are searched for with an exact integer test, not placed by hand.

Layout note: the plane sits left and the readout right, for the whole scene.
Moving the figure between beats to make room for text is the commonest way a
generated scene reads as stitched together rather than composed.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.plane import (distance, integer_points_on_circle, legs,
                                    verify_on_circle, verify_pythagoras)
from forge.primitives.vectors import Vec2

AX_C = GREY_B
PT_C = BLUE_C
ALT_C = "#F0AC5F"
DIST_C = "#5CD0B3"
DIM = GREY_B

UNIT = 0.6                   # screen units per grid step
PLANE_SHIFT = LEFT * 2.3     # leaves a readout column on the right


class CoordinatePlane(ForgeScene):

    def xy(self, x, y):
        """Grid coordinates to screen point — the single source of position."""
        return self.plane.c2p(x, y)

    def readout(self, *rows):
        """Stack lines in the right-hand column, top-aligned."""
        g = VGroup(*rows).arrange(DOWN, buff=0.34, aligned_edge=LEFT)
        g.move_to(RIGHT * 3.9).align_to(UP * 2.1, UP)
        return g

    @beat("Two lines, crossing at zero", seconds=8,
          narration="Take a number line, and stand a second one up through its "
                    "zero. That crossing point is the origin. Everything on "
                    "this page is now described by how far right you go, and "
                    "then how far up.")
    def axes(self):
        self.plane = NumberPlane(
            x_range=[-6, 6, 1], y_range=[-6, 6, 1],
            x_length=12 * UNIT, y_length=12 * UNIT,
            background_line_style={"stroke_color": GREY_E, "stroke_width": 1,
                                   "stroke_opacity": 0.55},
            # Tips on the axes read as arrows, and an arrow on a plane reads as
            # a vector. These are rulers, not vectors.
            axis_config={"include_tip": False, "stroke_color": AX_C,
                         "stroke_width": 2.2},
        ).shift(PLANE_SHIFT)

        self.title = Text("Two numbers locate a point",
                          font_size=30).to_edge(UP, buff=0.4)

        x_lab = MathTex("x", font_size=30, color=AX_C).next_to(
            self.xy(6, 0), RIGHT, buff=0.18)
        y_lab = MathTex("y", font_size=30, color=AX_C).next_to(
            self.xy(0, 6), UP, buff=0.18)
        origin = Dot(self.xy(0, 0), color=WHITE, radius=0.07)
        o_lab = MathTex("(0,0)", font_size=24, color=DIM).next_to(
            origin, DL, buff=0.12)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.plane.get_x_axis()), run_time=1.0)
        self.play(Create(self.plane.get_y_axis()), run_time=1.0)
        self.play(FadeIn(self.plane.background_lines), run_time=0.8)
        self.play(Write(x_lab), Write(y_lab), run_time=0.6)
        self.play(FadeIn(origin, scale=0.4), FadeIn(o_lab), run_time=0.6)
        self.axis_labels = VGroup(x_lab, y_lab, origin, o_lab)
        self.wait(0.8)

    @beat("A point needs two numbers, in that order", seconds=11,
          narration="Here is a point. To say where it is, walk three to the "
                    "right, then one up: three, one. Now swap them — one "
                    "right, three up — and you land somewhere else entirely. "
                    "The pair is ordered. Three one and one three are two "
                    "different places.")
    def address(self):
        p = Vec2(3.0, 1.0)
        q = Vec2(1.0, 3.0)

        dot_p = Dot(self.xy(*p.as_tuple()), color=PT_C, radius=0.09)
        # Guide lines are drawn to the axes from the dot's own position, so a
        # mislabelled coordinate would visibly miss its tick.
        gx = DashedLine(self.xy(p.x, 0), self.xy(*p.as_tuple()),
                        color=PT_C, stroke_width=2, dash_length=0.08)
        gy = DashedLine(self.xy(0, p.y), self.xy(*p.as_tuple()),
                        color=PT_C, stroke_width=2, dash_length=0.08)
        lab_p = MathTex(rf"({p.x:.0f},{p.y:.0f})", font_size=28, color=PT_C
                        ).next_to(dot_p, UR, buff=0.12)

        self.play(FadeIn(dot_p, scale=0.4), run_time=0.5)
        self.play(Create(gx), run_time=0.7)
        self.play(Create(gy), run_time=0.7)
        self.play(Write(lab_p), run_time=0.7)

        steps = self.readout(
            VGroup(Text("right", font_size=22, color=DIM),
                   MathTex(rf"{p.x:.0f}", font_size=32, color=PT_C)
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
            VGroup(Text("then up", font_size=22, color=DIM),
                   MathTex(rf"{p.y:.0f}", font_size=32, color=PT_C)
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
        )
        self.play(LaggedStart(*[FadeIn(s, shift=LEFT * 0.2) for s in steps],
                              lag_ratio=0.4, run_time=1.2))
        self.wait(0.8)

        dot_q = Dot(self.xy(*q.as_tuple()), color=ALT_C, radius=0.09)
        lab_q = MathTex(rf"({q.x:.0f},{q.y:.0f})", font_size=28, color=ALT_C
                        ).next_to(dot_q, UL, buff=0.12)
        swap = MathTex(rf"({p.x:.0f},{p.y:.0f})", r"\neq",
                       rf"({q.x:.0f},{q.y:.0f})", font_size=34)
        swap[0].set_color(PT_C)
        swap[2].set_color(ALT_C)
        swap.next_to(steps, DOWN, buff=0.7).align_to(steps, LEFT)

        self.play(FadeIn(dot_q, scale=0.4), Write(lab_q), run_time=0.8)
        self.play(Write(swap), run_time=1.1)
        self.wait(1.2)

        # Sweep everything but the plane: later beats reuse the same space, and
        # a leftover guide line would be read as part of the next argument.
        self.play(*[FadeOut(m) for m in
                    (dot_p, dot_q, gx, gy, lab_p, lab_q, steps, swap)],
                  run_time=0.7)

    @beat("Distance falls out of a right triangle", seconds=11,
          narration="Two points, and the gap between them. Going across is "
                    "three; going up is four. Those two legs and the straight "
                    "line between the points form a right triangle — so the "
                    "distance is the square root of three squared plus four "
                    "squared. Exactly five.")
    def distance_beat(self):
        a, b = Vec2(-2.0, -1.0), Vec2(1.0, 3.0)
        dx, dy = legs(a, b)
        d = distance(a, b)
        assert verify_pythagoras(a, b)       # checked before it is claimed

        da = Dot(self.xy(*a.as_tuple()), color=PT_C, radius=0.085)
        db = Dot(self.xy(*b.as_tuple()), color=PT_C, radius=0.085)
        la = MathTex(rf"({a.x:.0f},{a.y:.0f})", font_size=24, color=PT_C
                     ).next_to(da, DL, buff=0.1)
        lb = MathTex(rf"({b.x:.0f},{b.y:.0f})", font_size=24, color=PT_C
                     ).next_to(db, UR, buff=0.1)

        corner = self.xy(b.x, a.y)
        run = Line(self.xy(*a.as_tuple()), corner, color=ALT_C, stroke_width=4)
        rise = Line(corner, self.xy(*b.as_tuple()), color=ALT_C, stroke_width=4)
        hyp = Line(self.xy(*a.as_tuple()), self.xy(*b.as_tuple()),
                   color=DIST_C, stroke_width=5)
        run_l = MathTex(rf"{dx:.0f}", font_size=28, color=ALT_C
                        ).next_to(run, DOWN, buff=0.12)
        rise_l = MathTex(rf"{dy:.0f}", font_size=28, color=ALT_C
                         ).next_to(rise, RIGHT, buff=0.12)

        self.play(FadeIn(da, scale=0.4), FadeIn(db, scale=0.4), run_time=0.6)
        self.play(Write(la), Write(lb), run_time=0.7)
        self.play(Create(run), run_time=0.7)
        self.play(Write(run_l), run_time=0.4)
        self.play(Create(rise), run_time=0.7)
        self.play(Write(rise_l), run_time=0.4)
        self.play(Create(hyp), run_time=1.0)

        calc = self.readout(
            MathTex(rf"\sqrt{{{dx:.0f}^2 + {dy:.0f}^2}}", font_size=34,
                    color=ALT_C),
            MathTex(rf"= \sqrt{{{dx*dx + dy*dy:.0f}}}", font_size=34, color=DIM),
            MathTex(rf"= {d:.0f}", font_size=40, color=DIST_C),
        )
        self.play(LaggedStart(*[Write(c) for c in calc],
                              lag_ratio=0.55, run_time=2.2))
        self.wait(1.4)
        self.play(*[FadeOut(m) for m in
                    (da, db, la, lb, run, rise, hyp, run_l, rise_l, calc)],
                  run_time=0.7)

    @beat("A rule picks out a shape", seconds=13,
          narration="Now turn it around. Instead of naming one point, name a "
                    "rule: every point exactly five from the origin. Twelve of "
                    "them land on grid corners — three four, four three, five "
                    "zero, and their reflections. But the rule does not stop "
                    "at grid corners. Allow every point that satisfies it and "
                    "the answer is a circle. That is what a graph is: a rule, "
                    "drawn.")
    def locus(self):
        r = 5
        lattice = integer_points_on_circle(r)
        assert verify_on_circle(lattice, r)
        assert len(lattice) == 12

        rule = self.readout(
            MathTex(r"x^2 + y^2 = 25", font_size=34, color=DIST_C),
            Text("every point 5 from the origin", font_size=19, color=DIM),
        )
        self.play(Write(rule[0]), run_time=1.0)
        self.play(FadeIn(rule[1]), run_time=0.5)

        dots = VGroup(*[Dot(self.xy(*p.as_tuple()), color=ALT_C, radius=0.075)
                        for p in lattice])
        self.play(LaggedStart(*[FadeIn(d, scale=0.4) for d in dots],
                              lag_ratio=0.12, run_time=2.2))

        count = MathTex(rf"{len(lattice)}", r"\text{ whole-number points}",
                        font_size=26)
        count[0].set_color(ALT_C)
        count[1].set_color(DIM)
        count.next_to(rule, DOWN, buff=0.6).align_to(rule, LEFT)
        self.play(FadeIn(count, shift=UP * 0.12), run_time=0.8)
        self.wait(1.2)

        circle = Circle(radius=r * UNIT, color=DIST_C,
                        stroke_width=4).move_to(self.xy(0, 0))
        self.play(Create(circle), run_time=2.0)
        self.play(dots.animate.set_color(DIST_C), run_time=0.8)
        self.wait(2.0)
