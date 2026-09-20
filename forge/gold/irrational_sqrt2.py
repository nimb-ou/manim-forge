"""Gold scene 028 — why root two is not a fraction.

Number theory, Tier 2, and the corpus's first proof by contradiction. The
shape of the argument is the lesson: assume the thing, follow it honestly, and
arrive somewhere impossible.

The two halves are kept apart on purpose. Searching for a fraction that works
is *evidence* -- better and better approximations that never arrive -- and the
scene says so rather than implying a finite search settles anything. The proof
is the parity argument, and it is the only part that concludes.

``verify_no_exact_fraction`` tests in integers (n*n == 2*d*d), never in floats:
(n/d)**2 == 2.0 is a coin toss at the boundary and would eventually "find" a
solution that does not exist.
"""

import math

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.numbers import (parity_of_square, sqrt2_contradiction,
                                      verify_no_exact_fraction)

ROOT_C = "#5CD0B3"
TRY_C = "#F0AC5F"
BAD_C = "#FC6255"
DIM = GREY_B

CANDIDATES = [(3, 2), (7, 5), (17, 12), (99, 70), (1393, 985)]


class IrrationalSqrt2(ForgeScene):

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 3.7).align_to(UP * 2.2, UP)

    @beat("A length that obviously exists", seconds=19,
          narration="A square with sides of one. Its diagonal is a real "
                    "length — you can draw it, measure it, walk along it. By "
                    "Pythagoras it is the square root of two. The question is "
                    "whether that length can be written as one whole number "
                    "divided by another.")
    def the_length(self):
        self.title = Text("Why root two is not a fraction",
                          font_size=30).to_edge(UP, buff=0.4)
        s = 2.1
        at = LEFT * 3.9 + DOWN * 0.5
        self.square = Square(side_length=s, color=GREY_B, stroke_width=2.6,
                             fill_color=BLUE_E, fill_opacity=0.18).move_to(at)
        c = self.square.get_corner(DL)
        d = self.square.get_corner(UR)
        self.diag = Line(c, d, color=ROOT_C, stroke_width=4.5)
        one_b = MathTex("1", font_size=24, color=DIM).next_to(self.square, DOWN,
                                                              buff=0.16)
        one_l = MathTex("1", font_size=24, color=DIM).next_to(self.square, LEFT,
                                                              buff=0.16)
        dlab = MathTex(r"\sqrt{2}", font_size=28, color=ROOT_C).next_to(
            self.diag.point_from_proportion(0.55), UL, buff=0.08)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.square), run_time=1.1)
        self.play(Write(one_b), Write(one_l), run_time=0.7)
        self.play(Create(self.diag), run_time=1.0)
        self.play(Write(dlab), run_time=0.7)

        q = self.panel(
            MathTex(r"\sqrt{2} = \frac{n}{d}\;?", font_size=36, color=TRY_C),
            Text("for whole numbers n and d", font_size=18, color=DIM),
        )
        self.play(Write(q[0]), run_time=1.0)
        self.play(FadeIn(q[1]), run_time=0.5)
        self.marks = VGroup(one_b, one_l, dlab)
        self.q = q
        self.wait(1.2)

    @beat("Search for one, and get close but never there", seconds=21,
          narration="Try to find such a fraction. Three halves is close. Seven "
                    "fifths is closer. Ninety-nine seventieths is closer "
                    "still, and thirteen ninety-three over nine eighty-five "
                    "agrees to six decimal places. But not one of them is "
                    "exact, and no search however long could settle the "
                    "matter — it can only fail to find one.")
    def search(self):
        self.play(FadeOut(self.q), run_time=0.3)
        rows = VGroup()
        for n, d in CANDIDATES:
            err = abs(n / d - math.sqrt(2))
            rows.add(VGroup(
                MathTex(rf"\frac{{{n}}}{{{d}}}", font_size=26, color=TRY_C),
                MathTex(rf"{n/d:.9f}", font_size=20, color=DIM),
                MathTex(rf"{err:.1e}", font_size=18, color=BAD_C),
            ).arrange(RIGHT, buff=0.3))
        rows.arrange(DOWN, buff=0.26, aligned_edge=LEFT)
        rows.move_to(RIGHT * 3.7 + UP * 1.4)

        head = MathTex(rf"\sqrt{{2}} = {math.sqrt(2):.9f}\ldots", font_size=24,
                       color=ROOT_C).next_to(rows, UP, buff=0.4).align_to(rows, LEFT)
        self.play(Write(head), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.4, run_time=2.4))

        best_n, best_d, best_e = sqrt2_contradiction()
        assert (best_n, best_d) == CANDIDATES[-1]   # the closest under den 2000
        note = Text("close is not equal", font_size=20, color=BAD_C).next_to(
            rows, DOWN, buff=0.45).align_to(rows, LEFT)
        self.play(FadeIn(note, shift=UP * 0.12), run_time=0.8)
        self.search_block = VGroup(head, rows, note)
        self.wait(1.4)

    @beat("Assume it works, and follow it honestly", seconds=21,
          narration="So assume it does work. Root two equals n over d, with "
                    "the fraction already reduced — no common factor left. "
                    "Square both sides: n squared is two d squared. So n "
                    "squared is even. But an odd number squared is odd, so n "
                    "itself must be even. Write n as two k.")
    def assume(self):
        assert parity_of_square(3) == "odd" and parity_of_square(4) == "even"

        self.play(FadeOut(self.search_block), run_time=0.4)
        lines = VGroup(
            MathTex(r"\sqrt{2} = \frac{n}{d}", font_size=30, color=TRY_C),
            Text("already reduced", font_size=17, color=DIM),
            MathTex(r"2d^2 = n^2", font_size=30, color=TRY_C),
            MathTex(r"\Rightarrow n^2 \text{ is even}", font_size=26, color=DIM),
            MathTex(r"\Rightarrow n \text{ is even}", font_size=26, color=ROOT_C),
            MathTex(r"n = 2k", font_size=28, color=ROOT_C),
        ).arrange(DOWN, buff=0.26, aligned_edge=LEFT)
        lines.move_to(RIGHT * 3.5 + UP * 0.9)

        parity = VGroup(*[
            VGroup(MathTex(rf"{m}^2 = {m*m}", font_size=19, color=DIM),
                   Text(parity_of_square(m), font_size=15,
                        color=ROOT_C if m % 2 == 0 else BAD_C)
                   ).arrange(RIGHT, buff=0.18)
            for m in (3, 4, 5, 6)])
        parity.arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        parity.next_to(lines[4], LEFT, buff=0.9)

        for i, ln in enumerate(lines):
            self.play(Write(ln) if isinstance(ln, MathTex) else FadeIn(ln),
                      run_time=0.85)
            if i == 3:
                self.play(LaggedStart(*[FadeIn(p) for p in parity],
                                      lag_ratio=0.15, run_time=1.0))
        self.lines, self.parity = lines, parity
        self.wait(1.2)

    @beat("And arrive somewhere impossible", seconds=27,
          narration="Substitute. Two d squared is four k squared, so d squared "
                    "is two k squared — and by exactly the same argument, d is "
                    "even too. But n and d were both even, and the fraction "
                    "was supposed to be reduced. That is the contradiction. "
                    "The assumption was the only thing that could be wrong, so "
                    "no such fraction exists. Not a large one, not a clever "
                    "one. None.")
    def contradiction(self):
        assert verify_no_exact_fraction()

        self.play(FadeOut(self.parity), run_time=0.3)
        self.play(self.lines.animate.scale(0.8).to_edge(UP, buff=1.2).shift(RIGHT * 2.2),
                  run_time=0.9)

        more = VGroup(
            MathTex(r"2d^2 = (2k)^2 = 4k^2", font_size=28, color=TRY_C),
            MathTex(r"d^2 = 2k^2", font_size=28, color=TRY_C),
            MathTex(r"\Rightarrow d \text{ is even too}", font_size=26, color=ROOT_C),
        ).arrange(DOWN, buff=0.24, aligned_edge=LEFT)
        more.next_to(self.lines, DOWN, buff=0.45).align_to(self.lines, LEFT)
        for ln in more:
            self.play(Write(ln), run_time=0.9)

        clash = VGroup(
            Text("both even", font_size=22, color=BAD_C),
            Text("but the fraction was reduced", font_size=19, color=DIM),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        clash.next_to(more, DOWN, buff=0.45).align_to(more, LEFT)
        self.play(FadeIn(clash, shift=UP * 0.12), run_time=1.0)
        self.play(Indicate(clash[0], color=BAD_C, scale_factor=1.15),
                  run_time=0.9)

        verdict = Text("no such fraction exists", font_size=26,
                       color=ROOT_C).next_to(clash, DOWN, buff=0.45).align_to(
            clash, LEFT)
        self.play(Write(verdict), run_time=1.3)
        self.play(self.diag.animate.set_stroke(ROOT_C, width=6), run_time=0.8)
        self.wait(2.2)
