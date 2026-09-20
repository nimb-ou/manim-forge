"""Gold scene 014 — why atoms fill up in layers.

Chemistry, Tier 2, and the first chemistry scene in the corpus. That matters:
a model with no chemistry examples still answers chemistry prompts, and "two,
eight, eighteen" is exactly the kind of sequence a half-trained model renders
as "two, eight, ten" with total confidence and no visual tell.

So the scene derives the numbers instead of listing them. Each shell's
capacity is counted from the orbitals inside it -- 1, then 1+3, then 1+3+5 --
and the scene asserts the count equals 2n^2 before drawing it. The sum of the
first n odd numbers being n^2 is the whole reason the formula has a square in
it, and it is visible here rather than asserted.

The filling order is separately flagged as what it is: a fact about energies,
not about capacity. Shell three pauses at eight because 4s sits below 3d.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.chem import (fill_shells, orbitals_in_shell,
                                   shell_capacity, valence_electrons,
                                   verify_capacity_is_2n2)

NUC_C = "#FC6255"
E_C = BLUE_C
SHELL_C = GREY_C
HI_C = "#F0AC5F"
DIM = GREY_B

CENTRE = LEFT * 3.2
RADII = (0.85, 1.55, 2.25)


class ElectronShells(ForgeScene):

    def shell_ring(self, i):
        return Circle(radius=RADII[i], color=SHELL_C, stroke_width=1.6,
                      stroke_opacity=0.7).move_to(CENTRE)

    def electrons_on(self, i, k, colour=E_C):
        """``k`` electrons spaced evenly round shell ``i``."""
        r = RADII[i]
        return VGroup(*[
            Dot(CENTRE + r * np.array([np.cos(a), np.sin(a), 0.0]),
                color=colour, radius=0.075)
            for a in [2 * np.pi * j / k + np.pi / 2 for j in range(k)]])

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.36, aligned_edge=LEFT)
        return g.move_to(RIGHT * 3.4).align_to(UP * 2.4, UP)

    @beat("One nucleus, one electron", seconds=15,
          narration="Hydrogen: a nucleus, and one electron. The electron does "
                    "not sit still and it does not follow a track — but it "
                    "does keep to a certain distance, and that distance is "
                    "what we draw as a shell.")
    def hydrogen(self):
        self.title = Text("Why atoms fill up in layers",
                          font_size=30).to_edge(UP, buff=0.4)
        self.nucleus = Dot(CENTRE, color=NUC_C, radius=0.17)
        nlab = Text("nucleus", font_size=18, color=DIM).next_to(
            self.nucleus, DOWN, buff=0.22)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(FadeIn(self.nucleus, scale=0.4), FadeIn(nlab), run_time=0.8)

        ring = self.shell_ring(0)
        e = self.electrons_on(0, 1)
        self.play(Create(ring), run_time=0.9)
        self.play(FadeIn(e, scale=0.4), run_time=0.6)
        # A rotation about the nucleus keeps every electron at its own radius,
        # which is the one property of the picture that is actually true.
        self.play(e.animate.rotate(PI, about_point=CENTRE),
                  run_time=1.6, rate_func=linear)
        self.rings = VGroup(ring)
        self.shell_e = [e]
        self.nlab = nlab
        self.wait(0.8)

    @beat("The first shell holds two, and here is why", seconds=19,
          narration="The first shell has exactly one orbital — one shape the "
                    "electron can occupy. Two electrons fit in it, because an "
                    "electron has a spin that can point either of two ways, "
                    "and no two electrons in an atom may match completely. One "
                    "orbital, two spins, two electrons.")
    def shell_one(self):
        assert verify_capacity_is_2n2()
        n_orb = len(orbitals_in_shell(1))
        cap = shell_capacity(1)
        assert n_orb == 1 and cap == 2

        box = Square(side_length=0.5, color=HI_C, stroke_width=2.5)
        up = MathTex(r"\uparrow", font_size=30, color=E_C)
        down = MathTex(r"\downarrow", font_size=30, color=E_C)
        pair = VGroup(up, down).arrange(RIGHT, buff=0.12).move_to(box)

        self.count = self.panel(
            VGroup(MathTex(rf"n = 1", font_size=30, color=DIM),
                   ).arrange(RIGHT),
            VGroup(box, pair),
            VGroup(Text(f"{n_orb} orbital", font_size=21, color=DIM),
                   MathTex(r"\times", font_size=21, color=DIM),
                   Text("2 spins", font_size=21, color=DIM),
                   MathTex(rf"= {cap}", font_size=28, color=HI_C),
                   ).arrange(RIGHT, buff=0.22),
        )
        self.play(FadeIn(self.count[0]), run_time=0.5)
        self.play(Create(box), run_time=0.6)
        self.play(Write(up), run_time=0.4)

        second = self.electrons_on(0, 2)
        self.play(FadeOut(self.shell_e[0]), FadeIn(second), Write(down),
                  run_time=0.8)
        self.shell_e[0] = second
        self.play(FadeIn(self.count[2], shift=LEFT * 0.2), run_time=0.8)
        self.play(second.animate.rotate(PI, about_point=CENTRE),
                  run_time=1.6, rate_func=linear)
        self.wait(1.0)

    @beat("The second shell holds eight, for the same reason", seconds=13,
          narration="The second shell has more shapes available: one of the "
                    "first kind and three of the second. Four orbitals, two "
                    "electrons each — eight. Not a number to remember. A "
                    "number to count.")
    def shell_two(self):
        orbs = orbitals_in_shell(2)
        cap = shell_capacity(2)
        assert len(orbs) == 4 and cap == 8

        self.play(FadeOut(self.count), run_time=0.5)

        boxes = VGroup(*[Square(side_length=0.46, color=HI_C, stroke_width=2.2)
                         for _ in orbs]).arrange(RIGHT, buff=0.14)
        spins = VGroup(*[
            VGroup(MathTex(r"\uparrow", font_size=26, color=E_C),
                   MathTex(r"\downarrow", font_size=26, color=E_C)
                   ).arrange(RIGHT, buff=0.06).move_to(b)
            for b in boxes])

        groups = VGroup(
            VGroup(Text("l = 0", font_size=19, color=DIM),
                   Text("1 orbital", font_size=19, color=DIM)
                   ).arrange(RIGHT, buff=0.25),
            VGroup(Text("l = 1", font_size=19, color=DIM),
                   Text("3 orbitals", font_size=19, color=DIM)
                   ).arrange(RIGHT, buff=0.25),
        )
        self.count = self.panel(
            MathTex("n = 2", font_size=30, color=DIM),
            groups[0], groups[1],
            VGroup(boxes, spins),
            VGroup(MathTex(rf"{len(orbs)}", font_size=26, color=HI_C),
                   Text("orbitals, two each", font_size=20, color=DIM),
                   MathTex(rf"= {cap}", font_size=30, color=HI_C),
                   ).arrange(RIGHT, buff=0.22),
        )
        self.play(FadeIn(self.count[0]), run_time=0.4)
        self.play(FadeIn(groups[0], shift=LEFT * 0.2), run_time=0.5)
        self.play(FadeIn(groups[1], shift=LEFT * 0.2), run_time=0.5)
        self.play(LaggedStart(*[Create(b) for b in boxes],
                              lag_ratio=0.2, run_time=1.0))

        ring2 = self.shell_ring(1)
        e2 = self.electrons_on(1, cap)
        self.play(Create(ring2), run_time=0.7)
        self.play(LaggedStart(*[Write(s) for s in spins],
                              lag_ratio=0.15, run_time=1.2),
                  LaggedStart(*[FadeIn(d, scale=0.4) for d in e2],
                              lag_ratio=0.15, run_time=1.2))
        self.play(FadeIn(self.count[4], shift=LEFT * 0.2), run_time=0.7)
        self.play(e2.animate.rotate(PI / 2, about_point=CENTRE),
                  run_time=1.2, rate_func=linear)
        self.rings.add(ring2)
        self.shell_e.append(e2)
        self.wait(0.9)

    @beat("Odd numbers add to squares", seconds=21,
          narration="The orbital counts are one, four, nine — the squares. "
                    "They have to be: each shell adds the next odd number of "
                    "shapes, one then three then five, and odd numbers added "
                    "up always land on a square. Two electrons per orbital "
                    "gives two n squared. That is where the formula comes from.")
    def pattern(self):
        self.play(FadeOut(self.count), run_time=0.5)

        rows = VGroup()
        for n in (1, 2, 3):
            k = len(orbitals_in_shell(n))
            odds = " + ".join(str(2 * l + 1) for l in range(n))
            rows.add(VGroup(
                MathTex(rf"n={n}", font_size=25, color=DIM),
                MathTex(odds, font_size=25, color=HI_C),
                MathTex(rf"= {k} = {n}^2", font_size=25, color=DIM),
                MathTex(rf"\Rightarrow {shell_capacity(n)}", font_size=28, color=E_C),
            ).arrange(RIGHT, buff=0.26))
        rows.arrange(DOWN, buff=0.34, aligned_edge=LEFT)
        formula = MathTex(r"2n^2", font_size=44, color=E_C)
        self.count = self.panel(rows, formula)

        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.45, run_time=2.4))
        self.play(Write(formula), run_time=1.0)
        self.wait(1.8)

    @beat("A real atom, and the electron that does the chemistry", seconds=18,
          narration="Sodium has eleven electrons. Two fill the first shell, "
                    "eight fill the second, and one is left over, alone in the "
                    "third. That single outer electron is loosely held and "
                    "easily given away — which is the whole of sodium's "
                    "chemistry, read straight off the diagram.")
    def sodium(self):
        z = 11
        shells = fill_shells(z)
        assert shells == [2, 8, 1]
        val = valence_electrons(z)

        self.play(FadeOut(self.count), run_time=0.5)

        ring3 = self.shell_ring(2)
        e3 = self.electrons_on(2, shells[2], colour=HI_C)
        self.play(Create(ring3), run_time=0.7)
        self.play(FadeIn(e3, scale=0.4), run_time=0.6)

        rows = VGroup(*[
            VGroup(Text(f"shell {i+1}", font_size=21, color=DIM),
                   MathTex(rf"{k}", font_size=30,
                           color=HI_C if i == len(shells) - 1 else E_C),
                   ).arrange(RIGHT, buff=0.35, aligned_edge=DOWN)
            for i, k in enumerate(shells)])
        rows.arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        total = VGroup(Text("sodium", font_size=22, color=DIM),
                       MathTex(rf"Z = {z}", font_size=26, color=DIM)
                       ).arrange(RIGHT, buff=0.3)
        note = VGroup(
            MathTex(rf"{val}", font_size=32, color=HI_C),
            Text("outer electron", font_size=20, color=DIM),
        ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN)
        self.count = self.panel(total, rows, note)

        self.play(FadeIn(total), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.35, run_time=1.4))
        self.play(FadeIn(note, shift=UP * 0.15), run_time=0.7)
        self.play(Flash(e3[0], color=HI_C, line_length=0.18,
                        flash_radius=0.3, num_lines=10), run_time=1.0)
        self.wait(2.0)
