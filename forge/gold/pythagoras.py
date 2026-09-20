"""Gold scene 022 — Pythagoras by rearrangement.

Geometry, Tier 2. The proof is the one that needs no algebra: take two
identical big squares, put the same four triangles inside each in two
different ways, and whatever is left over must match. In one arrangement the
leftover is two squares on the legs; in the other it is one square on the
hypotenuse.

The triangle is 5-12-13 rather than 3-4-5. A 3-4-5 triangle is close enough to
half a square that the pinwheel arrangement reads as symmetric, which makes
the second picture look like a restatement of the first rather than a genuinely
different packing.

``verify_rearrangement`` checks that both packings really do leave the same
area and that it really is both a^2+b^2 and c^2, before the scene says so.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.geometry import RightTriangle

TRI_C = "#F0AC5F"
LEG_C = BLUE_C
HYP_C = "#5CD0B3"
FRAME_C = GREY_B
DIM = GREY_B

T = RightTriangle(a=5.0, b=12.0)
U = 0.22                       # screen units per triangle unit
LEFT_AT = LEFT * 3.4 + DOWN * 0.45
RIGHT_AT = RIGHT * 1.0 + DOWN * 0.45


class Pythagoras(ForgeScene):

    def square_at(self, centre):
        s = T.outer_square_side() * U
        return Square(side_length=s, color=FRAME_C, stroke_width=2.6).move_to(centre)

    def tri(self, pts, colour=TRI_C):
        return Polygon(*pts, color=colour, fill_color=colour,
                       fill_opacity=0.45, stroke_width=2)

    def corner(self, centre):
        s = T.outer_square_side() * U
        return centre + np.array([-s / 2, -s / 2, 0.0])

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.to_edge(RIGHT, buff=0.5).align_to(UP * 2.3, UP)

    @beat("One right triangle, and the squares on its sides", seconds=21,
          narration="A right triangle, with legs of five and twelve. Build a "
                    "square on each of the three sides. The claim to be "
                    "proved is that the two smaller squares, added together, "
                    "come to exactly the same area as the big one — and that "
                    "this holds for every right triangle, not just this one.")
    def setup(self):
        self.title = Text("Pythagoras, by rearrangement",
                          font_size=30).to_edge(UP, buff=0.4)

        a, b, c = T.a * U, T.b * U, T.c * U
        base = LEFT * 1.4 + DOWN * 1.6
        A = base
        B = base + RIGHT * b
        C = base + UP * a
        self.triangle = self.tri([A, B, C], HYP_C)

        la = MathTex(rf"{T.a:.0f}", font_size=24, color=LEG_C).next_to(
            Line(A, C), LEFT, buff=0.14)
        lb = MathTex(rf"{T.b:.0f}", font_size=24, color=LEG_C).next_to(
            Line(A, B), DOWN, buff=0.14)
        lc = MathTex(rf"{T.c:.0f}", font_size=24, color=HYP_C).next_to(
            Line(B, C).point_from_proportion(0.5), UR, buff=0.06)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.triangle), run_time=1.3)
        self.play(Write(la), Write(lb), Write(lc), run_time=1.1)

        sq_a = Square(side_length=a, color=LEG_C, fill_color=LEG_C,
                      fill_opacity=0.25, stroke_width=2).next_to(
            Line(A, C), LEFT, buff=0)
        sq_b = Square(side_length=b, color=LEG_C, fill_color=LEG_C,
                      fill_opacity=0.25, stroke_width=2).next_to(
            Line(A, B), DOWN, buff=0)
        self.play(FadeIn(sq_a, scale=0.7), FadeIn(sq_b, scale=0.7), run_time=1.2)

        claim = self.panel(
            MathTex(r"a^2 + b^2 = c^2", font_size=36, color=HYP_C),
            VGroup(MathTex(rf"{T.a:.0f}^2 + {T.b:.0f}^2", font_size=26, color=LEG_C),
                   MathTex(rf"= {T.a**2 + T.b**2:.0f}", font_size=26, color=DIM)
                   ).arrange(RIGHT, buff=0.2),
            VGroup(MathTex(rf"{T.c:.0f}^2", font_size=26, color=HYP_C),
                   MathTex(rf"= {T.c**2:.0f}", font_size=26, color=DIM)
                   ).arrange(RIGHT, buff=0.2),
        )
        self.play(Write(claim[0]), run_time=1.0)
        self.claim = claim
        self.small = VGroup(sq_a, sq_b, la, lb, lc)
        self.wait(1.4)

    @beat("Four copies, packed into a square, one way", seconds=20,
          narration="Take four identical copies of the triangle and a square "
                    "whose side is the two legs added together. Pack the four "
                    "into two opposite corners. What is left is two squares — "
                    "one with side a, one with side b. Nothing has been "
                    "computed yet; the pieces were simply put somewhere.")
    def packing_one(self):
        self.play(FadeOut(self.small), FadeOut(self.triangle), run_time=0.6)

        frame = self.square_at(LEFT_AT)
        o = self.corner(LEFT_AT)
        a, b = T.a * U, T.b * U
        # Two corners: a triangle on the bottom-left and one top-right, each
        # doubled, leaving an a-square and a b-square exposed.
        tris = VGroup(
            self.tri([o, o + RIGHT * b, o + RIGHT * b + UP * a]),
            self.tri([o, o + UP * (a + b), o + RIGHT * b + UP * a]),
            self.tri([o + RIGHT * (a + b), o + RIGHT * (a + b) + UP * (a + b),
                      o + RIGHT * b + UP * a]),
            self.tri([o + RIGHT * (a + b), o + RIGHT * b,
                      o + RIGHT * b + UP * a]),
        )
        lab = Text("two corners", font_size=20, color=DIM).next_to(frame, DOWN,
                                                                  buff=0.28)
        self.play(Create(frame), run_time=0.9)
        self.play(LaggedStart(*[FadeIn(t, scale=0.7) for t in tris],
                              lag_ratio=0.25, run_time=2.0))
        self.play(FadeIn(lab), run_time=0.5)
        self.frame_one, self.tris_one, self.lab_one = frame, tris, lab

        left = T.leftover_two_squares()
        note = self.panel(
            Text("left over", font_size=20, color=DIM),
            VGroup(MathTex(r"a^2 + b^2", font_size=30, color=LEG_C),
                   MathTex(rf"= {left:.0f}", font_size=28, color=DIM)
                   ).arrange(RIGHT, buff=0.2),
        )
        self.play(FadeOut(self.claim), run_time=0.3)
        self.play(LaggedStart(*[FadeIn(n, shift=LEFT * 0.2) for n in note],
                              lag_ratio=0.35, run_time=1.2))
        self.note_one = note
        self.wait(1.4)

    @beat("The same four, packed the other way", seconds=18,
          narration="Now the same square, and the same four triangles, turned "
                    "into a pinwheel around the edges. Every piece is "
                    "identical to before and the outer square is identical to "
                    "before. What is left this time is a single tilted square, "
                    "and its side is the hypotenuse.")
    def packing_two(self):
        frame = self.square_at(RIGHT_AT)
        o = self.corner(RIGHT_AT)
        a, b, s = T.a * U, T.b * U, (T.a + T.b) * U
        # Pinwheel: each triangle sits on one side, legs a and b alternating.
        tris = VGroup(
            self.tri([o, o + RIGHT * b, o + UP * a]),
            self.tri([o + RIGHT * b, o + RIGHT * s, o + RIGHT * s + UP * b]),
            self.tri([o + RIGHT * s + UP * b, o + RIGHT * s + UP * s,
                      o + RIGHT * a + UP * s]),
            self.tri([o + RIGHT * a + UP * s, o + UP * s, o + UP * a]),
        )
        lab = Text("pinwheel", font_size=20, color=DIM).next_to(frame, DOWN,
                                                               buff=0.28)
        self.play(Create(frame), run_time=0.9)
        self.play(LaggedStart(*[FadeIn(t, scale=0.7) for t in tris],
                              lag_ratio=0.25, run_time=2.0))
        self.play(FadeIn(lab), run_time=0.5)

        tilted = Polygon(o + RIGHT * b, o + RIGHT * s + UP * b,
                         o + RIGHT * a + UP * s, o + UP * a,
                         color=HYP_C, fill_color=HYP_C, fill_opacity=0.3,
                         stroke_width=2.5)
        self.play(FadeIn(tilted), run_time=1.0)
        self.frame_two, self.tris_two, self.tilted = frame, tris, tilted
        self.lab_two = lab
        self.wait(1.4)

    @beat("Same square, same pieces, so the leftovers match", seconds=24,
          narration="Both outer squares have the same area. Both contain the "
                    "same four triangles. So whatever is left must be equal — "
                    "and that is the proof. The two small squares on the legs "
                    "and the one tilted square on the hypotenuse are the same "
                    "area, here and for every right triangle, because nothing "
                    "in the argument used the particular numbers.")
    def conclude(self):
        assert T.verify_rearrangement()
        left = T.leftover_two_squares()
        assert abs(left - (T.a ** 2 + T.b ** 2)) < 1e-9
        assert abs(left - T.c ** 2) < 1e-9

        self.play(FadeOut(self.note_one), run_time=0.3)
        self.play(LaggedStart(
            *[t.animate.set_fill(opacity=0.75) for t in self.tris_one],
            *[t.animate.set_fill(opacity=0.75) for t in self.tris_two],
            lag_ratio=0.08, run_time=1.4))

        eq = VGroup(
            VGroup(MathTex(r"a^2 + b^2", font_size=30, color=LEG_C),
                   MathTex(r"=", font_size=30, color=DIM),
                   MathTex(r"c^2", font_size=30, color=HYP_C),
                   ).arrange(RIGHT, buff=0.25),
            VGroup(MathTex(rf"{T.a**2:.0f} + {T.b**2:.0f}", font_size=26, color=LEG_C),
                   MathTex(r"=", font_size=26, color=DIM),
                   MathTex(rf"{T.c**2:.0f}", font_size=26, color=HYP_C),
                   ).arrange(RIGHT, buff=0.25),
            Text("same pieces, same frame", font_size=19, color=DIM),
        ).arrange(DOWN, buff=0.28).to_edge(DOWN, buff=0.45)

        self.play(Write(eq[0]), run_time=1.3)
        self.play(FadeIn(eq[1], shift=UP * 0.12), run_time=0.8)
        self.play(FadeIn(eq[2]), run_time=0.6)
        self.wait(2.4)
