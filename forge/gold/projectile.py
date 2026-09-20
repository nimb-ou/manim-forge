"""Gold scene 033 — two motions at once.

Physics, Tier 2. The insight worth animating is that a projectile is not
doing something complicated; it is doing two simple things simultaneously.
Horizontally it moves at a constant speed forever. Vertically it is a ball
thrown straight up. The parabola is what those two look like from the side.

Every number comes from ``Shot``, and every trajectory is checked against a
step-by-step integration before it is drawn -- a parabola is trivially easy to
draw and impossible to check by eye, so the closed form is verified against
motion actually simulated under gravity.

The complementary-angle result is checked across the whole quadrant, not on
the pair the scene shows. Two arcs landing together in one picture is a
coincidence until it is shown to be a rule.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.motion import (G, Shot, best_angle, complementary,
                                     verify_complementary_same_range)

PATH_C = "#5CD0B3"
HORIZ_C = "#F0AC5F"
VERT_C = BLUE_C
ALT_C = "#8B7FD4"
DIM = GREY_B

SPEED = 20.0
ANGLE = 60.0


class Projectile(ForgeScene):

    def setup_axes(self):
        return Axes(
            x_range=[0, 45, 10], y_range=[0, 18, 5],
            x_length=8.0, y_length=3.6,
            axis_config={"include_tip": False, "stroke_color": GREY_B,
                         "stroke_width": 2, "font_size": 18},
        ).shift(LEFT * 1.3 + DOWN * 1.2)

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.4).align_to(UP * 2.3, UP)

    def arc(self, shot, colour, width=4):
        pts = [self.axes.c2p(x, y) for x, y in shot.path()]
        return VMobject(color=colour, stroke_width=width).set_points_smoothly(pts)

    @beat("Throw something, and it traces a curve", seconds=15,
          narration="Launch a ball at twenty metres per second, sixty degrees "
                    "above the horizontal. It arcs up, over, and down, landing "
                    "thirty-five metres away after three and a half seconds. "
                    "The curve looks like one complicated motion. It is not.")
    def launch(self):
        self.title = Text("A projectile is two motions at once",
                          font_size=29).to_edge(UP, buff=0.4)
        self.axes = self.setup_axes()
        self.shot = Shot(SPEED, ANGLE)
        assert self.shot.verify_closed_form()
        assert round(self.shot.range_m) == 35        # spoken
        assert round(self.shot.flight_time, 1) == 3.5

        xl = self.axes.get_x_axis_label(Text("metres", font_size=17, color=DIM),
                                        edge=DOWN, direction=DOWN, buff=0.25)
        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.axes), FadeIn(xl), run_time=1.3)

        self.path = self.arc(self.shot, PATH_C)
        ball = Dot(self.axes.c2p(0, 0), color=WHITE, radius=0.09)
        self.play(FadeIn(ball, scale=0.5), run_time=0.4)
        self.play(Create(self.path),
                  MoveAlongPath(ball, self.path), run_time=2.6,
                  rate_func=linear)

        facts = self.panel(
            VGroup(Text("speed", font_size=19, color=DIM),
                   MathTex(rf"{SPEED:.0f}\,\text{{m/s}}", font_size=24, color=DIM)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(Text("angle", font_size=19, color=DIM),
                   MathTex(rf"{ANGLE:.0f}^\circ", font_size=24, color=DIM)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(Text("range", font_size=19, color=DIM),
                   MathTex(rf"{self.shot.range_m:.1f}\,\text{{m}}", font_size=24,
                           color=PATH_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
        )
        self.play(LaggedStart(*[FadeIn(f, shift=LEFT * 0.2) for f in facts],
                              lag_ratio=0.3, run_time=1.6))
        self.ball, self.facts, self.xl = ball, facts, xl
        self.wait(1.2)

    @beat("Sideways: nothing happens to it at all", seconds=16,
          narration="Watch only the horizontal position. It advances by the "
                    "same amount every tick, start to finish. Gravity pulls "
                    "downwards and has no sideways component, so nothing ever "
                    "slows the ball horizontally. Ten metres per second, "
                    "constant, for the whole flight.")
    def horizontal(self):
        self.play(FadeOut(self.facts), run_time=0.3)
        T = self.shot.flight_time
        ticks = [T * i / 8 for i in range(9)]
        shadows = VGroup(*[
            Dot(self.axes.c2p(self.shot.at(t)[0], 0), color=HORIZ_C, radius=0.07)
            for t in ticks])
        drops = VGroup(*[
            DashedLine(self.axes.c2p(*self.shot.at(t)),
                       self.axes.c2p(self.shot.at(t)[0], 0),
                       color=HORIZ_C, stroke_width=1.6, dash_length=0.07)
            for t in ticks])

        self.play(LaggedStart(*[Create(d) for d in drops],
                              lag_ratio=0.12, run_time=1.6))
        self.play(LaggedStart(*[FadeIn(s, scale=0.5) for s in shadows],
                              lag_ratio=0.12, run_time=1.4))

        gaps = [self.shot.at(ticks[i + 1])[0] - self.shot.at(ticks[i])[0]
                for i in range(8)]
        assert max(gaps) - min(gaps) < 1e-9      # genuinely equal spacing

        note = self.panel(
            VGroup(MathTex(r"v_x", font_size=26, color=HORIZ_C),
                   MathTex(rf"= {self.shot.vx:.0f}\,\text{{m/s}}", font_size=24,
                           color=HORIZ_C)
                   ).arrange(RIGHT, buff=0.2),
            Text("every tick the same", font_size=19, color=DIM),
            Text("gravity has no sideways part", font_size=17, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(n, shift=LEFT * 0.2) for n in note],
                              lag_ratio=0.3, run_time=1.6))
        self.shadows, self.drops, self.hnote = shadows, drops, note
        self.wait(1.4)

    @beat("Upward: a ball thrown straight up", seconds=19,
          narration="Now only the height. It rises quickly, slows, stops for "
                    "an instant fifteen point three metres up, then falls back "
                    "faster and faster. That is exactly what a ball thrown "
                    "straight upward does — the same motion, with the sideways "
                    "drift removed. Two independent motions, sharing a clock.")
    def vertical(self):
        assert round(self.shot.apex, 1) == 15.3        # spoken in the narration
        apex = self.shot.apex

        self.play(FadeOut(self.hnote), FadeOut(self.drops),
                  FadeOut(self.shadows), run_time=0.5)
        T = self.shot.flight_time
        ticks = [T * i / 8 for i in range(9)]
        rail_x = 41.0
        rail = Line(self.axes.c2p(rail_x, 0), self.axes.c2p(rail_x, 17.5),
                    color=VERT_C, stroke_width=2.4)
        marks = VGroup(*[
            Dot(self.axes.c2p(rail_x, self.shot.at(t)[1]), color=VERT_C,
                radius=0.07) for t in ticks])
        links = VGroup(*[
            DashedLine(self.axes.c2p(*self.shot.at(t)),
                       self.axes.c2p(rail_x, self.shot.at(t)[1]),
                       color=VERT_C, stroke_width=1.4, dash_length=0.07)
            for t in ticks])

        self.play(Create(rail), run_time=0.7)
        self.play(LaggedStart(*[Create(l) for l in links],
                              lag_ratio=0.1, run_time=1.5))
        self.play(LaggedStart(*[FadeIn(m, scale=0.5) for m in marks],
                              lag_ratio=0.1, run_time=1.3))

        apex_line = DashedLine(self.axes.c2p(0, apex), self.axes.c2p(rail_x, apex),
                               color=VERT_C, stroke_width=1.8, dash_length=0.1)
        alab = MathTex(rf"{apex:.1f}\,\text{{m}}", font_size=22,
                       color=VERT_C).next_to(self.axes.c2p(0, apex), LEFT, buff=0.12)
        self.play(Create(apex_line), FadeIn(alab), run_time=1.0)

        note = self.panel(
            Text("bunched at the top", font_size=19, color=VERT_C),
            Text("spread at the ends", font_size=19, color=VERT_C),
            Text("a ball thrown straight up", font_size=18, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(n, shift=LEFT * 0.2) for n in note],
                              lag_ratio=0.3, run_time=1.5))
        self.vgroup = VGroup(rail, marks, links, apex_line, alab, note)
        self.wait(1.4)

    @beat("Two angles, one landing spot", seconds=25,
          narration="One consequence worth seeing. Fire at thirty degrees "
                    "instead of sixty and the ball lands in exactly the same "
                    "place — a flat fast shot and a high slow one, covering "
                    "identical ground. Any angle and its complement do this, "
                    "checked for every whole angle in the quadrant. Which "
                    "leaves forty-five degrees, the one angle that is its own "
                    "complement, as the longest throw.")
    def complement(self):
        assert verify_complementary_same_range(SPEED)
        other = Shot(SPEED, complementary(ANGLE))
        assert abs(other.range_m - self.shot.range_m) < 1e-9
        assert best_angle(SPEED) == 45.0

        self.play(FadeOut(self.vgroup), FadeOut(self.ball), run_time=0.5)
        low = self.arc(other, ALT_C)
        self.play(Create(low), run_time=1.8)

        land = Dot(self.axes.c2p(self.shot.range_m, 0), color=WHITE, radius=0.09)
        self.play(FadeIn(land, scale=0.4),
                  Flash(land, color=WHITE, line_length=0.18,
                        flash_radius=0.3, num_lines=10), run_time=1.0)

        rows = self.panel(
            VGroup(MathTex(rf"{ANGLE:.0f}^\circ", font_size=24, color=PATH_C),
                   MathTex(rf"{self.shot.range_m:.2f}\,\text{{m}}", font_size=22,
                           color=PATH_C)).arrange(RIGHT, buff=0.3),
            VGroup(MathTex(rf"{complementary(ANGLE):.0f}^\circ", font_size=24,
                           color=ALT_C),
                   MathTex(rf"{other.range_m:.2f}\,\text{{m}}", font_size=22,
                           color=ALT_C)).arrange(RIGHT, buff=0.3),
            Text("identical", font_size=19, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.3, run_time=1.6))

        best = Shot(SPEED, 45.0)
        top = self.arc(best, HORIZ_C, width=3)
        blab = VGroup(
            MathTex(r"45^\circ", font_size=26, color=HORIZ_C),
            Text("the longest", font_size=19, color=HORIZ_C),
        ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN)
        blab.next_to(rows, DOWN, buff=0.55).align_to(rows, LEFT)
        self.play(Create(top), run_time=1.4)
        self.play(FadeIn(blab, shift=UP * 0.12), run_time=0.8)
        self.wait(2.2)
