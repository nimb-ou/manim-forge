"""Gold scene 043 — the determinant is an area factor.

Vectors, Tier 2, building on the transform scene. The formula ad minus bc is
easy to memorise and tells you nothing. What it *is* is the factor by which
every area changes, and the sign is whether the plane got flipped over.

The scene measures rather than asserts. ``verify_det_is_area_factor`` maps two
polygons -- the unit square and a deliberately irregular quadrilateral -- and
compares their shoelace areas before and after. The second shape matters: a
check done only on an axis-aligned square would pass for maps that stretch
correctly and shear wrongly.

The zero case is the payoff. A determinant of zero means every area is
crushed to nothing, the whole plane collapses onto a line, and information is
destroyed -- which is exactly why such a matrix has no inverse.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.vectors import (Mat2, Vec2, polygon_area,
                                      verify_det_is_area_factor)

I_C = "#5CD0B3"
J_C = "#F0AC5F"
AREA_C = BLUE_C
FLIP_C = "#8B7FD4"
ZERO_C = "#FC6255"
DIM = GREY_B

M = Mat2(3.0, 1.0, 1.0, 2.0)          # det 5
FLIP = Mat2(1.0, 2.0, 2.0, 1.0)       # det -3
SQUASH = Mat2(2.0, 1.0, 4.0, 2.0)     # det 0
U = 0.72
CENTRE = LEFT * 2.0 + DOWN * 0.3
SHAPE = [Vec2(-0.4, -0.2), Vec2(1.3, 0.1), Vec2(0.9, 1.4), Vec2(-0.7, 0.8)]


class Determinant(ForgeScene):

    def p(self, v):
        return CENTRE + np.array([v.x * U, v.y * U, 0.0])

    def unit_square(self, m=None, colour=AREA_C):
        pts = [Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1)]
        if m:
            pts = [m.apply(p) for p in pts]
        return Polygon(*[self.p(p) for p in pts], color=colour,
                       fill_color=colour, fill_opacity=0.32, stroke_width=2.6)

    def blob(self, m=None, colour=FLIP_C):
        pts = [m.apply(p) for p in SHAPE] if m else SHAPE
        return Polygon(*[self.p(p) for p in pts], color=colour,
                       fill_color=colour, fill_opacity=0.3, stroke_width=2.2)

    def axes_lines(self):
        g = VGroup()
        for k in range(-6, 7):
            g.add(Line(self.p(Vec2(k, -6)), self.p(Vec2(k, 6)),
                       color=GREY_E, stroke_width=1.1, stroke_opacity=0.5))
            g.add(Line(self.p(Vec2(-6, k)), self.p(Vec2(6, k)),
                       color=GREY_E, stroke_width=1.1, stroke_opacity=0.5))
        return g

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.3).align_to(UP * 2.3, UP)

    @beat("Start with a square of area one", seconds=15,
          narration="The grid, and the square between the two basis vectors. "
                    "Its area is one, by definition — that square is the unit "
                    "the rest of the plane is measured in. Watch what a "
                    "transformation does to it.")
    def start(self):
        self.title = Text("The determinant is an area factor",
                          font_size=29).to_edge(UP, buff=0.4)
        self.grid = self.axes_lines()
        self.square = self.unit_square()

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.grid), run_time=1.4)
        self.play(FadeIn(self.square, scale=0.7), run_time=0.9)

        lab = self.panel(
            VGroup(Text("area", font_size=20, color=DIM),
                   MathTex("1", font_size=30, color=AREA_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            Text("the unit of measure", font_size=18, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(l, shift=LEFT * 0.2) for l in lab],
                              lag_ratio=0.3, run_time=1.2))
        self.lab = lab
        self.wait(1.2)

    @beat("Transform it, and measure what you get", seconds=17,
          narration="Apply a map. The square becomes a slanted parallelogram, "
                    "and measuring it gives an area of five. Not approximately "
                    "five — exactly, and the same five you get from ad minus "
                    "bc: three times two, minus one times one.")
    def measure(self):
        assert verify_det_is_area_factor(M)
        after = polygon_area([M.apply(p) for p in
                              (Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1))])
        assert abs(after - M.det) < 1e-12 and abs(M.det - 5.0) < 1e-12

        self.play(FadeOut(self.lab), run_time=0.3)
        self.play(Transform(self.square, self.unit_square(M)), run_time=2.2)

        rows = self.panel(
            VGroup(Text("measured area", font_size=19, color=DIM),
                   MathTex(rf"{after:.0f}", font_size=28, color=AREA_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            MathTex(rf"ad - bc = {M.a:.0f}\cdot{M.d:.0f} - {M.b:.0f}\cdot{M.c:.0f}",
                    font_size=24, color=DIM),
            MathTex(rf"= {M.det:.0f}", font_size=30, color=AREA_C),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.3, run_time=1.8))
        self.rows = rows
        self.wait(1.4)

    @beat("It is the same factor for every shape", seconds=21,
          narration="And it is not special to the square. Take any shape at "
                    "all — this lopsided quadrilateral, chosen precisely "
                    "because it is not aligned to anything — and its area is "
                    "multiplied by the same five. That is what the determinant "
                    "is: one number saying how much the map stretches every "
                    "area there is.")
    def any_shape(self):
        before = abs(polygon_area(SHAPE))
        after = abs(polygon_area([M.apply(p) for p in SHAPE]))
        assert abs(after / before - abs(M.det)) < 1e-9

        self.play(FadeOut(self.rows), FadeOut(self.square), run_time=0.4)
        self.play(Transform(self.grid, self.axes_lines()), run_time=0.01)
        blob = self.blob()
        self.play(FadeIn(blob, scale=0.8), run_time=0.9)
        self.play(Transform(blob, self.blob(M)), run_time=2.2)

        rows = self.panel(
            VGroup(Text("before", font_size=19, color=DIM),
                   MathTex(rf"{before:.3f}", font_size=24, color=FLIP_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(Text("after", font_size=19, color=DIM),
                   MathTex(rf"{after:.3f}", font_size=24, color=FLIP_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(Text("ratio", font_size=19, color=DIM),
                   MathTex(rf"{after/before:.3f}", font_size=28, color=AREA_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.3, run_time=1.6))
        self.blob_m, self.rows = blob, rows
        self.wait(1.4)

    @beat("Negative means flipped, zero means crushed", seconds=28,
          narration="Two cases are worth naming. A negative determinant means "
                    "the plane was turned over — the area is still scaled, but "
                    "left and right have swapped. And a determinant of zero "
                    "means every area became nothing: the whole plane has been "
                    "crushed onto a single line. Two different points now sit "
                    "in the same place, so there is no way back. That is why a "
                    "matrix with zero determinant has no inverse.")
    def special(self):
        assert FLIP.det < 0 and abs(SQUASH.det) < 1e-12
        assert verify_det_is_area_factor(FLIP)
        # The squash genuinely collapses: two distinct points land together.
        a, b = Vec2(1.0, -2.0), Vec2(0.0, 0.0)
        assert abs(SQUASH.apply(a).x - SQUASH.apply(b).x) < 1e-12
        assert abs(SQUASH.apply(a).y - SQUASH.apply(b).y) < 1e-12

        self.play(FadeOut(self.rows), FadeOut(self.blob_m), run_time=0.4)
        sq = self.unit_square()
        self.play(FadeIn(sq, scale=0.7), run_time=0.6)
        self.play(Transform(sq, self.unit_square(FLIP, colour=FLIP_C)),
                  Transform(self.grid, self.axes_lines()), run_time=2.0)
        flip_row = self.panel(
            VGroup(Text("det", font_size=19, color=DIM),
                   MathTex(rf"{FLIP.det:.0f}", font_size=30, color=FLIP_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            Text("flipped over", font_size=20, color=FLIP_C),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in flip_row],
                              lag_ratio=0.3, run_time=1.2))
        self.wait(1.2)

        self.play(FadeOut(flip_row), run_time=0.3)
        self.play(Transform(sq, self.unit_square(SQUASH, colour=ZERO_C)),
                  run_time=2.0)
        line = Line(self.p(Vec2(-2.4, -4.8)), self.p(Vec2(2.4, 4.8)),
                    color=ZERO_C, stroke_width=4)
        self.play(Create(line), run_time=1.0)

        zero_row = self.panel(
            VGroup(Text("det", font_size=19, color=DIM),
                   MathTex("0", font_size=32, color=ZERO_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            Text("area becomes nothing", font_size=19, color=ZERO_C),
            Text("two points, one place", font_size=18, color=DIM),
            Text("so: no inverse", font_size=21, color=ZERO_C),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in zero_row],
                              lag_ratio=0.3, run_time=1.8))
        self.wait(2.2)
