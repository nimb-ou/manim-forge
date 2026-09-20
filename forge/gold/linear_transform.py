"""Gold scene 042 — a matrix moves the whole plane.

Vectors, Tier 2. A matrix taught as a grid of numbers with a multiplication
rule is a thing to memorise. A matrix taught as a record of where the basis
vectors land is a thing you can see, and every property follows from it.

The scene checks that claim rather than asserting it: ``verify_columns_are_images``
confirms the columns really are the images of (1,0) and (0,1), and the grid
shown is built by applying the map to real lattice points, so a mis-stated
matrix would visibly disagree with its own columns.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.vectors import Mat2, Vec2, verify_columns_are_images

GRID_C = GREY_E
I_C = "#5CD0B3"
J_C = "#F0AC5F"
V_C = BLUE_C
DIM = GREY_B

M = Mat2(2.0, 1.0, 0.0, 3.0)
U = 0.78
CENTRE = LEFT * 1.9 + DOWN * 0.2
PROBE = Vec2(1.5, 0.5)


class LinearTransform(ForgeScene):

    def p(self, v):
        return CENTRE + np.array([v.x * U, v.y * U, 0.0])

    def grid_lines(self, m=None, colour=GRID_C, width=1.4, span=5):
        """The integer grid, optionally after the map. Real lattice points."""
        g = VGroup()
        f = (lambda v: m.apply(v)) if m else (lambda v: v)
        for k in range(-span, span + 1):
            row = [f(Vec2(x / 2, k)) for x in range(-2 * span, 2 * span + 1)]
            col = [f(Vec2(k, y / 2)) for y in range(-2 * span, 2 * span + 1)]
            for pts in (row, col):
                ln = VMobject(color=colour, stroke_width=width,
                              stroke_opacity=0.75)
                ln.set_points_as_corners([self.p(v) for v in pts])
                g.add(ln)
        return g

    def arrow(self, v, colour, width=4.5):
        return Arrow(self.p(Vec2(0, 0)), self.p(v), buff=0, color=colour,
                     stroke_width=width, max_tip_length_to_length_ratio=0.22)

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.3).align_to(UP * 2.3, UP)

    @beat("The plane, and the two vectors that describe it", seconds=23,
          narration="The whole plane, with its grid. Two special vectors: one "
                    "step right, and one step up. Every other point is built "
                    "from those two — a point at one and a half, nought point "
                    "five, is one and a half of the first plus half of the "
                    "second. Nothing else is needed to name any point at all.")
    def plane(self):
        self.title = Text("A matrix moves the whole plane",
                          font_size=30).to_edge(UP, buff=0.4)
        self.grid = self.grid_lines()
        self.i_hat = self.arrow(Vec2(1, 0), I_C)
        self.j_hat = self.arrow(Vec2(0, 1), J_C)
        self.v = self.arrow(PROBE, V_C, width=4)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.grid), run_time=2.0)
        self.play(GrowArrow(self.i_hat), GrowArrow(self.j_hat), run_time=1.0)
        self.play(GrowArrow(self.v), run_time=0.8)

        note = self.panel(
            VGroup(MathTex(r"\hat{\imath}", font_size=26, color=I_C),
                   Text("one right", font_size=18, color=DIM)
                   ).arrange(RIGHT, buff=0.25),
            VGroup(MathTex(r"\hat{\jmath}", font_size=26, color=J_C),
                   Text("one up", font_size=18, color=DIM)
                   ).arrange(RIGHT, buff=0.25),
            MathTex(rf"v = {PROBE.x}\,\hat{{\imath}} + {PROBE.y}\,\hat{{\jmath}}",
                    font_size=24, color=V_C),
        )
        self.play(LaggedStart(*[FadeIn(n, shift=LEFT * 0.2) for n in note],
                              lag_ratio=0.3, run_time=1.6))
        self.note = note
        self.wait(1.2)

    @beat("Move only those two, and everything follows", seconds=19,
          narration="Now move them. Send the first to two right and nothing "
                    "up; send the second to one right and three up. Insist on "
                    "one thing only: that the grid stays a grid — evenly "
                    "spaced, parallel lines, origin fixed. That single "
                    "demand forces where every other point must go.")
    def transform(self):
        assert verify_columns_are_images(M)
        c1, c2 = M.columns

        self.play(FadeOut(self.note), run_time=0.3)
        target_grid = self.grid_lines(M)
        self.play(Transform(self.grid, target_grid),
                  Transform(self.i_hat, self.arrow(c1, I_C)),
                  Transform(self.j_hat, self.arrow(c2, J_C)),
                  Transform(self.v, self.arrow(M.apply(PROBE), V_C, width=4)),
                  run_time=3.4)

        note = self.panel(
            VGroup(MathTex(r"\hat{\imath} \to", font_size=24, color=I_C),
                   MathTex(rf"({c1.x:.0f}, {c1.y:.0f})", font_size=24, color=I_C)
                   ).arrange(RIGHT, buff=0.2),
            VGroup(MathTex(r"\hat{\jmath} \to", font_size=24, color=J_C),
                   MathTex(rf"({c2.x:.0f}, {c2.y:.0f})", font_size=24, color=J_C)
                   ).arrange(RIGHT, buff=0.2),
            Text("grid stays a grid", font_size=19, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(n, shift=LEFT * 0.2) for n in note],
                              lag_ratio=0.3, run_time=1.6))
        self.note = note
        self.wait(1.4)

    @beat("The matrix is just those two landing places", seconds=20,
          narration="Write the two destinations side by side as columns, and "
                    "that is the matrix. Not a grid of numbers with a rule "
                    "bolted on — a record of where the two basis vectors went. "
                    "The multiplication rule everybody learns is only the "
                    "bookkeeping for rebuilding a point from its two pieces.")
    def matrix(self):
        c1, c2 = M.columns
        self.play(FadeOut(self.note), run_time=0.3)

        mat = Matrix([[f"{M.a:.0f}", f"{M.b:.0f}"],
                      [f"{M.c:.0f}", f"{M.d:.0f}"]],
                     h_buff=1.0, element_alignment_corner=ORIGIN)
        mat.scale(0.9).move_to(RIGHT * 4.2 + UP * 1.7)
        cols = mat.get_columns()
        cols[0].set_color(I_C)
        cols[1].set_color(J_C)
        self.play(Write(mat), run_time=1.6)

        braces = VGroup(
            Text("where i went", font_size=16, color=I_C),
            Text("where j went", font_size=16, color=J_C),
        ).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        braces.next_to(mat, DOWN, buff=0.4)
        self.play(FadeIn(braces, shift=UP * 0.12), run_time=0.9)

        work = VGroup(
            MathTex(rf"{PROBE.x}\,({c1.x:.0f},{c1.y:.0f}) + "
                    rf"{PROBE.y}\,({c2.x:.0f},{c2.y:.0f})",
                    font_size=22, color=DIM),
            MathTex(rf"= ({M.apply(PROBE).x:.1f}, {M.apply(PROBE).y:.1f})",
                    font_size=26, color=V_C),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        work.next_to(braces, DOWN, buff=0.5).align_to(braces, LEFT)
        self.play(Write(work[0]), run_time=1.2)
        self.play(FadeIn(work[1], shift=UP * 0.12), run_time=0.8)
        self.mat_group = VGroup(mat, braces, work)
        self.wait(1.4)

    @beat("Two maps in a row are one map", seconds=22,
          narration="And because a map is fully described by where the two "
                    "vectors land, doing one map and then another is itself "
                    "just a map — you only need to ask where the two end up "
                    "after both. That is what multiplying matrices means. Not "
                    "a rule about rows and columns; a question about where two "
                    "arrows finish.")
    def compose(self):
        N = Mat2(0.0, -1.0, 1.0, 0.0)          # quarter turn
        NM = N @ M
        # Composition must agree with applying the two in order, at real points.
        for pt in (Vec2(1, 0), Vec2(0, 1), PROBE, Vec2(-1.3, 2.1)):
            got = NM.apply(pt)
            want = N.apply(M.apply(pt))
            assert abs(got.x - want.x) < 1e-12 and abs(got.y - want.y) < 1e-12

        self.play(FadeOut(self.mat_group), run_time=0.4)
        c1, c2 = NM.columns
        self.play(Transform(self.grid, self.grid_lines(NM)),
                  Transform(self.i_hat, self.arrow(c1, I_C)),
                  Transform(self.j_hat, self.arrow(c2, J_C)),
                  Transform(self.v, self.arrow(NM.apply(PROBE), V_C, width=4)),
                  run_time=2.6)

        rows = self.panel(
            Text("then a quarter turn", font_size=19, color=DIM),
            VGroup(MathTex(r"\hat{\imath} \to", font_size=22, color=I_C),
                   MathTex(rf"({c1.x:.0f}, {c1.y:.0f})", font_size=22, color=I_C)
                   ).arrange(RIGHT, buff=0.2),
            VGroup(MathTex(r"\hat{\jmath} \to", font_size=22, color=J_C),
                   MathTex(rf"({c2.x:.0f}, {c2.y:.0f})", font_size=22, color=J_C)
                   ).arrange(RIGHT, buff=0.2),
            Text("one map, not two", font_size=20, color=V_C),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.3, run_time=1.8))
        self.wait(2.2)
