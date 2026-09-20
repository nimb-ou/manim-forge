"""Gold scene 027 — Euclid's algorithm as shrinking rectangles.

Number theory, Tier 2. The algebra (repeatedly replace the pair by the smaller
number and the remainder) is easy to state and impossible to feel. The
geometry makes it obvious: an a-by-b rectangle, tiled with the largest squares
that fit, leaves a smaller rectangle -- and the last square to fit exactly is
the greatest common divisor.

Steps come from ``euclid``; the quotients on screen are the quotients the
algorithm computes, and ``verify_euclid`` checks the answer against brute-force
divisor search for every pair under 120 before the scene names one.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.numbers import euclid, verify_euclid

RECT_C = GREY_B
SQ_C = [BLUE_C, "#F0AC5F", "#5CD0B3", "#FC6255", "#8B7FD4"]
GCD_C = "#5CD0B3"
DIM = GREY_B

A, B = 1071, 462
U = 0.0062                    # screen units per integer unit
ORIGIN_AT = LEFT * 4.6 + DOWN * 1.6


class EuclidGcd(ForgeScene):

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 3.4).align_to(UP * 2.3, UP)

    @beat("A rectangle whose sides are the two numbers", seconds=17,
          narration="Two numbers: one thousand and seventy-one, and four "
                    "hundred and sixty-two. What is the largest number that "
                    "divides both? Draw them as a rectangle, that many units "
                    "wide and that many tall. A common divisor is then a "
                    "square that tiles the rectangle exactly.")
    def rectangle(self):
        self.title = Text("Euclid's algorithm, drawn",
                          font_size=30).to_edge(UP, buff=0.4)

        self.rect = Rectangle(width=A * U, height=B * U, color=RECT_C,
                              stroke_width=2.5)
        self.rect.move_to(ORIGIN_AT + np.array([A * U / 2, B * U / 2, 0.0]))
        wlab = MathTex(rf"{A}", font_size=24, color=DIM).next_to(self.rect, DOWN,
                                                                 buff=0.18)
        hlab = MathTex(rf"{B}", font_size=24, color=DIM).next_to(self.rect, LEFT,
                                                                 buff=0.18)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.rect), run_time=1.4)
        self.play(Write(wlab), Write(hlab), run_time=0.8)

        q = self.panel(
            Text("largest square", font_size=20, color=DIM),
            Text("that tiles it exactly", font_size=20, color=GCD_C),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in q],
                              lag_ratio=0.3, run_time=1.2))
        self.labels = VGroup(wlab, hlab)
        self.q = q
        self.wait(1.4)

    @beat("Cut off as many squares as will fit", seconds=22,
          narration="Take the largest square that fits: four hundred and "
                    "sixty-two on a side. Two of them fit along the length, "
                    "and a strip is left over, one hundred and forty-seven "
                    "wide. Any square that tiled the original must also tile "
                    "that strip — so the problem has got smaller without the "
                    "answer changing at all.")
    def first_cut(self):
        assert verify_euclid()
        self.steps = euclid(A, B)
        s0 = self.steps[0]
        assert (s0.q, s0.r) == (2, 147)        # spoken in the narration

        self.play(FadeOut(self.q), run_time=0.3)
        x = ORIGIN_AT[0]
        squares = VGroup()
        for k in range(s0.q):
            sq = Square(side_length=s0.b * U, color=SQ_C[0], fill_color=SQ_C[0],
                        fill_opacity=0.3, stroke_width=2)
            sq.move_to(np.array([x + s0.b * U / 2,
                                 ORIGIN_AT[1] + s0.b * U / 2, 0.0]))
            squares.add(sq)
            x += s0.b * U
        self.play(LaggedStart(*[FadeIn(sq, scale=0.85) for sq in squares],
                              lag_ratio=0.35, run_time=1.6))

        strip = Rectangle(width=s0.r * U, height=s0.b * U, color=SQ_C[1],
                          stroke_width=2.5)
        strip.move_to(np.array([x + s0.r * U / 2,
                                ORIGIN_AT[1] + s0.b * U / 2, 0.0]))
        self.play(Create(strip), run_time=0.9)

        row = VGroup(
            MathTex(rf"{s0.a} = {s0.q}\times{s0.b} + {s0.r}", font_size=26,
                    color=SQ_C[1]),
        )
        self.rows = VGroup(row).arrange(DOWN, buff=0.26, aligned_edge=LEFT)
        self.rows.move_to(RIGHT * 3.4 + UP * 1.9)
        self.play(Write(row), run_time=1.2)
        self.tiles = VGroup(squares, strip)
        self.wait(1.4)

    @beat("Repeat on what is left, until nothing is", seconds=19,
          narration="Now do the same to the strip. Squares of one hundred and "
                    "forty-seven: three fit, and twenty-one is left. Squares "
                    "of twenty-one: seven fit, and nothing is left over. That "
                    "is the stopping point — the first square that tiles its "
                    "rectangle exactly.")
    def repeat(self):
        s = self.steps
        assert (s[1].q, s[1].r) == (3, 21)
        assert (s[2].q, s[2].r) == (7, 0)

        for i, st in enumerate(s[1:], start=1):
            row = MathTex(rf"{st.a} = {st.q}\times{st.b} + {st.r}",
                          font_size=26,
                          color=SQ_C[min(i + 1, len(SQ_C) - 1)]
                          if st.r else GCD_C)
            self.rows.add(row)
            self.rows.arrange(DOWN, buff=0.26, aligned_edge=LEFT)
            self.rows.move_to(RIGHT * 3.4 + UP * 1.9)
            self.play(Write(row), run_time=1.1)
            self.wait(0.5)

        final = VGroup(
            Text("remainder zero", font_size=19, color=DIM),
            Text("so it divides evenly", font_size=19, color=GCD_C),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        final.next_to(self.rows, DOWN, buff=0.5).align_to(self.rows, LEFT)
        self.play(FadeIn(final, shift=UP * 0.12), run_time=0.9)
        self.final = final
        self.wait(1.2)

    @beat("The last square is the answer", seconds=17,
          narration="Twenty-one. It divides both original numbers, and nothing "
                    "larger does — which the scene confirms by testing every "
                    "divisor up to the smaller number rather than taking "
                    "Euclid's word for it. Three steps, where checking every "
                    "candidate would have taken four hundred and sixty-two.")
    def answer(self):
        g = self.steps[-1].b
        assert g == 21
        assert A % g == 0 and B % g == 0
        assert all(not (A % d == 0 and B % d == 0)
                   for d in range(g + 1, min(A, B) + 1))

        self.play(FadeOut(self.final), run_time=0.3)
        box = VGroup(
            VGroup(MathTex(r"\gcd", font_size=26, color=DIM),
                   MathTex(rf"({A}, {B})", font_size=24, color=DIM)
                   ).arrange(RIGHT, buff=0.12),
            MathTex(rf"= {g}", font_size=44, color=GCD_C),
        ).arrange(DOWN, buff=0.24, aligned_edge=LEFT)
        box.next_to(self.rows, DOWN, buff=0.55).align_to(self.rows, LEFT)
        self.play(Write(box[0]), run_time=0.9)
        self.play(FadeIn(box[1], shift=UP * 0.15), run_time=0.9)

        cost = VGroup(
            VGroup(Text("Euclid", font_size=19, color=DIM),
                   MathTex(rf"{len(self.steps)}", font_size=26, color=GCD_C),
                   Text("steps", font_size=17, color=DIM)
                   ).arrange(RIGHT, buff=0.2),
            VGroup(Text("try every divisor", font_size=19, color=DIM),
                   MathTex(rf"{B}", font_size=26, color="#FC6255"),
                   ).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        cost.next_to(box, DOWN, buff=0.5).align_to(box, LEFT)
        self.play(LaggedStart(*[FadeIn(c, shift=LEFT * 0.2) for c in cost],
                              lag_ratio=0.35, run_time=1.4))
        self.wait(2.2)
