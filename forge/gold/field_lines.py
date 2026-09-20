"""Gold scene 039 — a field is a direction at every point.

Physics, Tier 2. Field lines are usually drawn as decoration around two
blobs, which teaches the picture and not the idea. The idea is that there is
an arrow at every point in space, and the lines are simply what you get by
walking along those arrows.

So the scene builds the lines that way. ``trace_line`` starts near a charge
and steps along the *unit* field direction until it leaves or arrives. The
often-repeated rule that field lines never cross is then not a drawing
convention but a consequence: the sum of the contributions has one value at
each point, so there is one direction to walk in.

Stepping along the unit direction rather than the raw vector matters. A line
traced with magnitude races near the charges and crawls far away, which reads
as a mistake even to a viewer who could not say why.
"""

import math

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.fields import (Charge, field_at, trace_line,
                                     verify_field_lines_do_not_cross)

POS_C = "#FC6255"
NEG_C = BLUE_C
ARROW_C = GREY_B
LINE_C = "#5CD0B3"
DIM = GREY_B

CHARGES = [Charge(-1.7, 0.0, 1.0), Charge(1.7, 0.0, -1.0)]
SC = 1.15
CENTRE = LEFT * 1.4 + DOWN * 0.2


class FieldLines(ForgeScene):

    def p(self, x, y):
        return CENTRE + np.array([x * SC, y * SC, 0.0])

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.6).align_to(UP * 2.3, UP)

    @beat("An arrow at every point", seconds=22,
          narration="Two charges: one positive, one negative. The field is not "
                    "the charges and it is not the lines you have seen drawn "
                    "around them. It is a rule that gives an arrow at every "
                    "single point in the space between — which way a small "
                    "positive test charge would be pushed if you put one "
                    "there.")
    def arrows(self):
        self.title = Text("A field is a direction at every point",
                          font_size=28).to_edge(UP, buff=0.4)

        self.dots = VGroup(
            Dot(self.p(CHARGES[0].x, CHARGES[0].y), color=POS_C, radius=0.16),
            Dot(self.p(CHARGES[1].x, CHARGES[1].y), color=NEG_C, radius=0.16))
        signs = VGroup(
            MathTex("+", font_size=30, color=WHITE).move_to(self.dots[0]),
            MathTex("-", font_size=32, color=WHITE).move_to(self.dots[1]))

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(FadeIn(self.dots, scale=0.5), FadeIn(signs), run_time=0.9)

        grid = VGroup()
        for gx in np.arange(-3.6, 3.7, 0.72):
            for gy in np.arange(-2.2, 2.3, 0.72):
                if any(math.hypot(gx - c.x, gy - c.y) < 0.55 for c in CHARGES):
                    continue
                ex, ey = field_at(CHARGES, gx, gy)
                m = math.hypot(ex, ey)
                if m < 1e-9:
                    continue
                d = np.array([ex / m, ey / m, 0.0]) * 0.3
                a = Arrow(self.p(gx, gy) - d * 0.5, self.p(gx, gy) + d * 0.5,
                          buff=0, color=ARROW_C, stroke_width=2,
                          max_tip_length_to_length_ratio=0.35,
                          stroke_opacity=0.75)
                grid.add(a)

        self.play(LaggedStart(*[GrowArrow(a) for a in grid],
                              lag_ratio=0.012, run_time=2.6))
        self.grid, self.signs = grid, signs

        note = self.panel(
            Text("one arrow", font_size=21, color=ARROW_C),
            Text("per point", font_size=21, color=ARROW_C),
            Text("not a picture of lines", font_size=17, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(n, shift=LEFT * 0.2) for n in note],
                              lag_ratio=0.3, run_time=1.4))
        self.note = note
        self.wait(1.2)

    @beat("Walk along the arrows and you get a line", seconds=20,
          narration="Now start somewhere and simply walk. Take a small step in "
                    "the direction the arrow points, look at the new arrow, "
                    "step again. Keep going. The curve that traces out is a "
                    "field line — not drawn, walked. Every one of them leaves "
                    "the positive charge and arrives at the negative one.")
    def walk(self):
        self.play(FadeOut(self.note), run_time=0.3)
        self.play(self.grid.animate.set_stroke(opacity=0.28), run_time=0.6)

        starts = []
        for ang in np.linspace(0.18, TAU - 0.18, 11):
            starts.append((CHARGES[0].x + 0.26 * math.cos(ang),
                           CHARGES[0].y + 0.26 * math.sin(ang)))

        self.lines = VGroup()
        for i, s in enumerate(starts):
            pts = trace_line(CHARGES, s)
            if len(pts) < 8:
                continue
            m = VMobject(color=LINE_C, stroke_width=2.6)
            m.set_points_smoothly([self.p(x, y) for x, y in pts])
            self.lines.add(m)

        self.play(Create(self.lines[len(self.lines) // 2]), run_time=1.6)
        self.wait(0.5)
        rest = VGroup(*[l for j, l in enumerate(self.lines)
                        if j != len(self.lines) // 2])
        self.play(LaggedStart(*[Create(l) for l in rest],
                              lag_ratio=0.12, run_time=3.0))

        note = self.panel(
            Text("step along the arrow", font_size=20, color=LINE_C),
            Text("look again, step again", font_size=20, color=LINE_C),
            Text("the line is the walk", font_size=18, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(n, shift=LEFT * 0.2) for n in note],
                              lag_ratio=0.3, run_time=1.4))
        self.note = note
        self.wait(1.4)

    @beat("Why they cannot cross", seconds=23,
          narration="This explains the rule everyone is told and nobody is "
                    "shown. Field lines never cross. Not because it would look "
                    "wrong, but because a crossing would mean two different "
                    "directions at one point — and the arrow at a point is one "
                    "vector, got by adding up the pulls from every charge. "
                    "There is only ever one answer.")
    def no_crossing(self):
        assert verify_field_lines_do_not_cross(CHARGES)
        self.play(FadeOut(self.note), run_time=0.3)

        probe = (0.0, 1.05)
        contribs = []
        for c, col in zip(CHARGES, (POS_C, NEG_C)):
            dx, dy = probe[0] - c.x, probe[1] - c.y
            r = math.hypot(dx, dy)
            s = 0.62 * c.q / (r * r)
            contribs.append((np.array([s * dx / r, s * dy / r, 0.0]), col))

        base = self.p(*probe)
        dot = Dot(base, color=WHITE, radius=0.08)
        self.play(FadeIn(dot, scale=0.5), run_time=0.5)

        arrows = VGroup(*[
            Arrow(base, base + v, buff=0, color=col, stroke_width=3.2,
                  max_tip_length_to_length_ratio=0.22)
            for v, col in contribs])
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows],
                              lag_ratio=0.35, run_time=1.4))

        total = contribs[0][0] + contribs[1][0]
        sum_arrow = Arrow(base, base + total, buff=0, color=LINE_C,
                          stroke_width=4.5, max_tip_length_to_length_ratio=0.2)
        self.play(GrowArrow(sum_arrow), run_time=1.0)

        note = self.panel(
            Text("add the pulls", font_size=20, color=DIM),
            Text("one vector results", font_size=21, color=LINE_C),
            Text("so one direction to walk", font_size=19, color=LINE_C),
            Text("crossing would need two", font_size=18, color=POS_C),
        )
        self.play(LaggedStart(*[FadeIn(n, shift=LEFT * 0.2) for n in note],
                              lag_ratio=0.3, run_time=1.8))
        self.probe_group = VGroup(dot, arrows, sum_arrow)
        self.note = note
        self.wait(1.6)

    @beat("Turn one charge around and the picture changes entirely", seconds=24,
          narration="One last thing, to show the lines really do come from the "
                    "arithmetic. Make the second charge positive as well. "
                    "Nothing about the method changes — same walk, same steps "
                    "— but now the two push against each other, the lines "
                    "refuse to join, and a point appears halfway between where "
                    "the field cancels exactly and there is no direction at "
                    "all.")
    def repel(self):
        both = [Charge(-1.7, 0.0, 1.0), Charge(1.7, 0.0, 1.0)]
        ex, ey = field_at(both, 0.0, 0.0)
        assert math.hypot(ex, ey) < 1e-12        # a genuine null point

        self.play(FadeOut(self.probe_group), FadeOut(self.note),
                  FadeOut(self.lines), FadeOut(self.grid), run_time=0.6)
        new_sign = MathTex("+", font_size=30, color=WHITE).move_to(self.dots[1])
        self.play(self.dots[1].animate.set_color(POS_C),
                  FadeTransform(self.signs[1], new_sign), run_time=0.8)

        grid = VGroup()
        for gx in np.arange(-3.6, 3.7, 0.72):
            for gy in np.arange(-2.2, 2.3, 0.72):
                if any(math.hypot(gx - c.x, gy - c.y) < 0.55 for c in both):
                    continue
                fx, fy = field_at(both, gx, gy)
                m = math.hypot(fx, fy)
                if m < 1e-9:
                    continue
                d = np.array([fx / m, fy / m, 0.0]) * 0.3
                grid.add(Arrow(self.p(gx, gy) - d * 0.5, self.p(gx, gy) + d * 0.5,
                               buff=0, color=ARROW_C, stroke_width=2,
                               max_tip_length_to_length_ratio=0.35,
                               stroke_opacity=0.4))
        self.play(LaggedStart(*[GrowArrow(a) for a in grid],
                              lag_ratio=0.01, run_time=1.8))

        lines = VGroup()
        for c in both:
            for ang in np.linspace(0.2, TAU - 0.2, 9):
                pts = trace_line(both, (c.x + 0.26 * math.cos(ang),
                                        c.y + 0.26 * math.sin(ang)))
                if len(pts) < 8:
                    continue
                m = VMobject(color=LINE_C, stroke_width=2.4)
                m.set_points_smoothly([self.p(x, y) for x, y in pts])
                lines.add(m)
        self.play(LaggedStart(*[Create(l) for l in lines],
                              lag_ratio=0.05, run_time=2.6))

        null = Dot(self.p(0, 0), color="#F0AC5F", radius=0.09)
        nlab = Text("field is exactly zero here", font_size=17,
                    color="#F0AC5F").next_to(null, DOWN, buff=0.28)
        self.play(FadeIn(null, scale=0.4), FadeIn(nlab), run_time=1.0)
        self.play(Flash(null, color="#F0AC5F", line_length=0.16,
                        flash_radius=0.3, num_lines=10), run_time=0.9)
        self.wait(2.2)
