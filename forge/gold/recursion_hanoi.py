"""Gold scene 026 — the tower, and why recursion is not a trick.

Discrete, Tier 2. Hanoi is usually shown as a puzzle being solved. The point
worth making is structural: the solution for n discs is the solution for n-1
discs, twice, with one move in between -- and that sentence is the whole
algorithm.

Moves come from the recursion itself. ``verify_hanoi`` replays every move onto
real pegs up to twelve discs and rejects any that puts a larger disc on a
smaller one, because a shorter, illegal sequence would still look like a
solution in a drawing.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.algorithms import hanoi, verify_hanoi

PEG_C = GREY_C
DISC_C = [BLUE_C, "#F0AC5F", "#5CD0B3", "#FC6255"]
HI_C = "#F0AC5F"
DIM = GREY_B

N_DISCS = 4
PEG_X = {"A": -4.3, "B": -1.9, "C": 0.5}
BASE_Y = -2.3
DISC_H = 0.34


class TowerOfHanoi(ForgeScene):

    def disc_width(self, d):
        return 0.55 + 0.34 * d

    def slot(self, peg, level):
        return np.array([PEG_X[peg], BASE_Y + DISC_H * (level + 0.5), 0.0])

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 3.9).align_to(UP * 2.3, UP)

    @beat("Three pegs, four discs, one rule", seconds=17,
          narration="Three pegs and four discs, largest at the bottom. Move "
                    "the whole stack to the far peg, one disc at a time, and "
                    "never place a larger disc on a smaller one. That last "
                    "rule is the only thing that makes this hard.")
    def setup(self):
        self.title = Text("The tower, and what recursion buys",
                          font_size=29).to_edge(UP, buff=0.4)

        pegs = VGroup(*[
            Line(np.array([x, BASE_Y, 0.0]), np.array([x, BASE_Y + 2.0, 0.0]),
                 color=PEG_C, stroke_width=4) for x in PEG_X.values()])
        base = Line(np.array([PEG_X["A"] - 1.0, BASE_Y, 0.0]),
                    np.array([PEG_X["C"] + 1.0, BASE_Y, 0.0]),
                    color=PEG_C, stroke_width=5)
        names = VGroup(*[
            Text(k, font_size=19, color=DIM).next_to(
                np.array([x, BASE_Y, 0.0]), DOWN, buff=0.22)
            for k, x in PEG_X.items()])

        self.stacks = {"A": list(range(N_DISCS, 0, -1)), "B": [], "C": []}
        self.discs = {}
        group = VGroup()
        for level, d in enumerate(self.stacks["A"]):
            r = RoundedRectangle(width=self.disc_width(d), height=DISC_H * 0.86,
                                 corner_radius=0.07, color=DISC_C[d - 1],
                                 fill_color=DISC_C[d - 1], fill_opacity=0.75,
                                 stroke_width=2).move_to(self.slot("A", level))
            self.discs[d] = r
            group.add(r)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(base), *[Create(p) for p in pegs], FadeIn(names),
                  run_time=1.3)
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.3) for r in reversed(group)],
                              lag_ratio=0.2, run_time=1.4))

        rule = self.panel(
            Text("never larger on smaller", font_size=21, color=HI_C),
            Text("one disc at a time", font_size=21, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rule],
                              lag_ratio=0.3, run_time=1.2))
        self.rule = rule
        self.wait(1.2)

    @beat("The idea: move everything above, then the big one", seconds=26,
          narration="Here is the whole idea, and it is one sentence. To move "
                    "four discs from A to C: move the top three out of the way "
                    "onto B, move the big one across, then move those three "
                    "on top of it. You do not have to know how to move three "
                    "discs — you only have to notice that it is the same "
                    "problem, one size smaller.")
    def idea(self):
        self.play(FadeOut(self.rule), run_time=0.3)
        steps = self.panel(
            VGroup(MathTex(r"1.", font_size=22, color=DIM),
                   Text("move 3 to B", font_size=21, color=DISC_C[2])
                   ).arrange(RIGHT, buff=0.22),
            VGroup(MathTex(r"2.", font_size=22, color=DIM),
                   Text("move disc 4 to C", font_size=21, color=DISC_C[3])
                   ).arrange(RIGHT, buff=0.22),
            VGroup(MathTex(r"3.", font_size=22, color=DIM),
                   Text("move 3 onto it", font_size=21, color=DISC_C[2])
                   ).arrange(RIGHT, buff=0.22),
        )
        self.play(LaggedStart(*[FadeIn(s, shift=LEFT * 0.2) for s in steps],
                              lag_ratio=0.4, run_time=1.8))

        top3 = VGroup(*[self.discs[d] for d in (1, 2, 3)])
        brace = Brace(top3, LEFT, color=HI_C, buff=0.14)
        blab = Text("the same problem", font_size=17, color=HI_C).next_to(
            brace, LEFT, buff=0.12)
        self.play(GrowFromCenter(brace), FadeIn(blab), run_time=1.0)
        self.wait(1.4)
        self.play(FadeOut(brace), FadeOut(blab), run_time=0.5)
        self.steps = steps

    @beat("Play it out, move by move", seconds=19,
          narration="Run it. Every move here comes from that one sentence "
                    "applied over and over, never from a plan anyone wrote "
                    "down. The big disc crosses exactly once, in the middle, "
                    "and the three smaller ones rebuild themselves on top of "
                    "it.")
    def play_out(self):
        assert verify_hanoi()
        moves = hanoi(N_DISCS, "A", "C", "B")
        assert len(moves) == 2 ** N_DISCS - 1

        counter = MathTex("0", font_size=30, color=DIM).next_to(
            self.steps, DOWN, buff=0.6).align_to(self.steps, LEFT)
        clab = Text("moves", font_size=18, color=DIM).next_to(counter, RIGHT,
                                                              buff=0.22)
        self.play(FadeIn(counter), FadeIn(clab), run_time=0.5)

        for i, m in enumerate(moves, start=1):
            disc = self.discs[m.disc]
            self.stacks[m.frm].pop()
            level = len(self.stacks[m.to])
            self.stacks[m.to].append(m.disc)
            lift = np.array([disc.get_center()[0], BASE_Y + 2.35, 0.0])
            over = np.array([PEG_X[m.to], BASE_Y + 2.35, 0.0])
            rt = 0.42 if m.disc != N_DISCS else 0.75
            self.play(disc.animate.move_to(lift), run_time=rt * 0.35)
            self.play(disc.animate.move_to(over), run_time=rt * 0.4)
            self.play(disc.animate.move_to(self.slot(m.to, level)),
                      run_time=rt * 0.35)
            new = MathTex(str(i), font_size=30,
                          color=HI_C if i == len(moves) else DIM).move_to(counter)
            self.play(FadeTransform(counter, new), run_time=0.14)
            counter = new
        self.counter, self.clab = counter, clab
        self.n_moves = len(moves)
        self.wait(1.0)

    @beat("Why the count doubles with every disc", seconds=24,
          narration="Fifteen moves. And the count follows from the sentence "
                    "too: solving n discs costs twice the cost of n minus "
                    "one, plus the single move in the middle. That doubling "
                    "gives two to the n, minus one. Four discs, fifteen "
                    "moves. Ten discs, a thousand and twenty-three. Sixty-four "
                    "discs, more moves than there have been seconds since the "
                    "universe began.")
    def count(self):
        assert self.n_moves == 15            # spoken
        assert len(hanoi(10)) == 1023        # spoken

        self.play(FadeOut(self.steps), FadeOut(self.counter), FadeOut(self.clab),
                  run_time=0.5)
        rows = VGroup()
        for n in (1, 2, 3, 4):
            rows.add(VGroup(
                MathTex(rf"{n}", font_size=22, color=DIM),
                Text("discs", font_size=16, color=DIM),
                MathTex(rf"{len(hanoi(n))}", font_size=26, color=HI_C),
            ).arrange(RIGHT, buff=0.22))
        rows.arrange(DOWN, buff=0.24, aligned_edge=LEFT)

        law = MathTex(r"T(n) = 2\,T(n-1) + 1", font_size=30, color=DISC_C[0])
        closed = MathTex(r"= 2^n - 1", font_size=32, color=HI_C)
        big = VGroup(
            MathTex(r"64", font_size=24, color=DIM),
            Text("discs", font_size=16, color=DIM),
            MathTex(rf"{2**64 - 1:.3e}".replace("e+", r"\times 10^{") + "}",
                    font_size=22, color=DISC_C[3]),
        ).arrange(RIGHT, buff=0.22)

        panel = self.panel(rows, law, closed, big)
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.3, run_time=1.6))
        self.play(Write(law), run_time=1.2)
        self.play(FadeIn(closed, shift=UP * 0.12), run_time=0.8)
        self.play(FadeIn(big), run_time=0.8)
        self.wait(2.4)
