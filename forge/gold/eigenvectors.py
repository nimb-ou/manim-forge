"""Gold scene 044 — the directions that do not turn.

Vectors, Tier 3. Almost every vector gets knocked off its line by a
transformation. A few special ones do not: they end up longer or shorter, or
pointing backwards, but still on the same line through the origin. Those are
the eigenvectors, and the factor is the eigenvalue.

``eigen2`` solves the characteristic polynomial and ``verify_eigen`` checks
that each returned pair genuinely satisfies Av = lambda v -- because a scene
that draws two arrows and calls them eigenvectors is unfalsifiable by eye.

The rotation at the end matters more than the successful case. It has no real
eigenvectors at all, ``eigen2`` returns an empty list, and the scene asserts
that emptiness. A demonstration that always finds two is drawing something
that is not there.
"""

import math

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.vectors import Mat2, Vec2, eigen2, verify_eigen

GRID_C = GREY_E
MOVED_C = GREY_B
EIG_C = "#5CD0B3"
EIG2_C = "#F0AC5F"
SPIN_C = "#FC6255"
DIM = GREY_B

M = Mat2(3.0, 1.0, 0.0, 2.0)
ROT = Mat2(0.0, -1.0, 1.0, 0.0)
U = 0.82
CENTRE = LEFT * 2.0 + DOWN * 0.2


class Eigenvectors(ForgeScene):

    def p(self, v):
        return CENTRE + np.array([v.x * U, v.y * U, 0.0])

    def arrow(self, v, colour, width=3.4, opacity=1.0):
        return Arrow(self.p(Vec2(0, 0)), self.p(v), buff=0, color=colour,
                     stroke_width=width, stroke_opacity=opacity,
                     max_tip_length_to_length_ratio=0.24)

    def ray(self, v, colour, k=5.0):
        return Line(self.p(Vec2(-v.x * k, -v.y * k)),
                    self.p(Vec2(v.x * k, v.y * k)),
                    color=colour, stroke_width=2, stroke_opacity=0.55)

    def grid(self):
        g = VGroup()
        for k in range(-5, 6):
            g.add(Line(self.p(Vec2(k, -5)), self.p(Vec2(k, 5)),
                       color=GRID_C, stroke_width=1.1, stroke_opacity=0.45))
            g.add(Line(self.p(Vec2(-5, k)), self.p(Vec2(5, k)),
                       color=GRID_C, stroke_width=1.1, stroke_opacity=0.45))
        return g

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.3).align_to(UP * 2.3, UP)

    @beat("Most directions get knocked off their line", seconds=16,
          narration="A transformation, and a fan of vectors pointing every "
                    "which way. Apply the map and watch what happens to the "
                    "directions. Almost all of them swing — they come out "
                    "pointing somewhere new, off the line they started on.")
    def fan(self):
        self.title = Text("The directions that do not turn",
                          font_size=30).to_edge(UP, buff=0.4)
        self.g = self.grid()
        dirs = [Vec2(math.cos(a), math.sin(a))
                for a in np.linspace(0, math.pi, 13)[:-1]]
        self.arrows = VGroup(*[self.arrow(v * 2.0, MOVED_C, 2.6, 0.8)
                               for v in dirs])

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.g), run_time=1.2)
        self.play(LaggedStart(*[GrowArrow(a) for a in self.arrows],
                              lag_ratio=0.08, run_time=1.8))

        moved = VGroup(*[self.arrow(M.apply(v * 2.0), MOVED_C, 2.6, 0.8)
                         for v in dirs])
        self.play(Transform(self.arrows, moved), run_time=2.4)
        self.dirs = dirs

        note = self.panel(
            Text("almost all swing", font_size=20, color=MOVED_C),
            Text("off their own line", font_size=20, color=MOVED_C),
        )
        self.play(LaggedStart(*[FadeIn(n, shift=LEFT * 0.2) for n in note],
                              lag_ratio=0.3, run_time=1.2))
        self.note = note
        self.wait(1.2)

    @beat("A few come out on the same line", seconds=21,
          narration="But not all. Two directions here come out pointing along "
                    "exactly the line they started on. One is stretched to "
                    "three times its length, the other to twice. They did not "
                    "turn at all — the map only scaled them. Those two "
                    "directions are the eigenvectors, and the scaling factors "
                    "are the eigenvalues.")
    def find(self):
        assert verify_eigen(M)
        pairs = eigen2(M)
        assert len(pairs) == 2
        lams = sorted(l for l, _ in pairs)
        assert lams == [2.0, 3.0]              # spoken in the narration

        self.play(FadeOut(self.arrows), FadeOut(self.note), run_time=0.5)
        colours = [EIG_C, EIG2_C]
        self.eig_group = VGroup()
        rows = VGroup()
        for (lam, v), col in zip(sorted(pairs, key=lambda p: -p[0]), colours):
            ray = self.ray(v, col)
            before = self.arrow(v * 1.6, col, 3.6)
            self.play(Create(ray), run_time=0.7)
            self.play(GrowArrow(before), run_time=0.6)
            after = self.arrow(M.apply(v * 1.6), col, 4.2)
            self.play(Transform(before, after), run_time=1.2)
            self.eig_group.add(ray, before)
            rows.add(VGroup(
                MathTex(rf"\lambda = {lam:.0f}", font_size=24, color=col),
                MathTex(rf"({v.x:.2f}, {v.y:.2f})", font_size=20, color=DIM),
            ).arrange(RIGHT, buff=0.3))
            self.wait(0.4)

        rows.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        rows.move_to(RIGHT * 4.3 + UP * 1.7)
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.35, run_time=1.4))
        self.rows, self.pairs = rows, pairs
        self.wait(1.4)

    @beat("Which is exactly what the equation says", seconds=20,
          narration="Written down, that is A v equals lambda v: the map "
                    "applied to the vector gives the same thing as simply "
                    "scaling the vector. Both sides land on the same point, "
                    "which the scene checks numerically before claiming it. "
                    "Finding these directions means asking which vectors the "
                    "map treats as mere scaling.")
    def equation(self):
        lam, v = max(self.pairs, key=lambda p: p[0])
        av = M.apply(v)
        assert abs(av.x - lam * v.x) < 1e-12 and abs(av.y - lam * v.y) < 1e-12

        self.play(FadeOut(self.rows), run_time=0.3)
        eq = MathTex(r"A\,\vec{v}", r"=", r"\lambda\,\vec{v}",
                     font_size=40)
        eq[0].set_color(EIG_C)
        eq[2].set_color(EIG_C)
        eq.move_to(RIGHT * 4.2 + UP * 1.8)
        self.play(Write(eq), run_time=1.4)

        check = VGroup(
            VGroup(MathTex(r"A\vec{v}", font_size=22, color=DIM),
                   MathTex(rf"= ({av.x:.3f}, {av.y:.3f})", font_size=21,
                           color=EIG_C)).arrange(RIGHT, buff=0.2),
            VGroup(MathTex(rf"{lam:.0f}\vec{{v}}", font_size=22, color=DIM),
                   MathTex(rf"= ({lam*v.x:.3f}, {lam*v.y:.3f})", font_size=21,
                           color=EIG_C)).arrange(RIGHT, buff=0.2),
            Text("same point", font_size=19, color=EIG_C),
        ).arrange(DOWN, buff=0.24, aligned_edge=LEFT)
        check.next_to(eq, DOWN, buff=0.55).align_to(eq, LEFT)
        self.play(LaggedStart(*[FadeIn(c, shift=LEFT * 0.2) for c in check],
                              lag_ratio=0.3, run_time=1.6))
        self.eq_group = VGroup(eq, check)
        self.wait(1.4)

    @beat("Sometimes there are none at all", seconds=30,
          narration="And sometimes there are none. Turn the plane by a quarter "
                    "circle. Every direction without exception comes out "
                    "pointing somewhere else — there is no line the rotation "
                    "leaves alone, so there is no real eigenvector to find. "
                    "Solving for one gives a negative under the square root. "
                    "That is not a failure of the method; it is the correct "
                    "answer, and a demonstration that always produces two "
                    "arrows is drawing something that is not there.")
    def none(self):
        assert eigen2(ROT) == []
        tr, det = ROT.a + ROT.d, ROT.det
        disc = tr * tr - 4 * det
        assert disc < 0                        # spoken: negative under the root

        self.play(FadeOut(self.eq_group), FadeOut(self.eig_group), run_time=0.5)
        dirs = [Vec2(math.cos(a), math.sin(a))
                for a in np.linspace(0, TAU, 17)[:-1]]
        arrows = VGroup(*[self.arrow(v * 2.0, SPIN_C, 2.6, 0.85) for v in dirs])
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows],
                              lag_ratio=0.05, run_time=1.6))
        self.play(Transform(arrows,
                            VGroup(*[self.arrow(ROT.apply(v * 2.0), SPIN_C,
                                                2.6, 0.85) for v in dirs])),
                  run_time=2.4)

        rows = self.panel(
            Text("a quarter turn", font_size=20, color=SPIN_C),
            MathTex(r"\lambda^2 + 1 = 0", font_size=28, color=SPIN_C),
            VGroup(Text("discriminant", font_size=18, color=DIM),
                   MathTex(rf"{disc:.0f}", font_size=24, color=SPIN_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            Text("no real eigenvector", font_size=21, color=SPIN_C),
            Text("and that is the answer", font_size=18, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.3, run_time=2.2))
        self.wait(2.2)
