"""Gold scene 029 — the door problem, played out.

Puzzles, Tier 2. Arguments about this puzzle are almost always arguments about
which game is being played, so the scene fixes the rules on screen before
showing anything: the host knows where the car is, and always opens a losing
door that the contestant did not pick.

``play_monty`` enforces exactly that, and the games shown are games the
simulation played. The two thirds is counted from twenty thousand of them, not
asserted -- and ``verify_monty_is_two_thirds`` also checks that the switch and
stay rates sum to one, which is the test that catches a host who opens at
random and quietly answers a different question.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.probability import (monty_rates, play_monty,
                                          verify_monty_is_two_thirds)

DOOR_C = GREY_B
PICK_C = "#F0AC5F"
CAR_C = "#5CD0B3"
GOAT_C = "#FC6255"
DIM = GREY_B

TRIALS = 20000
DOOR_W, DOOR_H = 1.35, 2.0
DOOR_X = [-4.6, -2.9, -1.2]
DOOR_Y = 0.35


class MontyHall(ForgeScene):

    def door(self, i):
        return Rectangle(width=DOOR_W, height=DOOR_H, color=DOOR_C,
                         stroke_width=2.6, fill_color=BLACK, fill_opacity=1.0
                         ).move_to(np.array([DOOR_X[i], DOOR_Y, 0.0]))

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 2.6).align_to(UP * 2.3, UP)

    @beat("Three doors, and the rules stated plainly", seconds=22,
          narration="Three doors. Behind one is a car, behind the other two, "
                    "goats. You pick a door. Then — and this is the part every "
                    "argument about this puzzle turns on — the host, who knows "
                    "where the car is, opens one of the doors you did not "
                    "pick, and always opens a goat. Should you switch?")
    def rules(self):
        self.title = Text("The door problem", font_size=30).to_edge(UP, buff=0.4)
        self.doors = VGroup(*[self.door(i) for i in range(3)])
        nums = VGroup(*[
            Text(str(i + 1), font_size=22, color=DIM).next_to(d, DOWN, buff=0.2)
            for i, d in enumerate(self.doors)])

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(LaggedStart(*[Create(d) for d in self.doors],
                              lag_ratio=0.2, run_time=1.4))
        self.play(FadeIn(nums), run_time=0.5)

        rules = self.panel(
            Text("the host knows", font_size=21, color=PICK_C),
            Text("always opens a goat", font_size=21, color=PICK_C),
            Text("never your door", font_size=21, color=PICK_C),
            Text("switch or stay?", font_size=23, color=CAR_C),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rules],
                              lag_ratio=0.3, run_time=2.0))
        self.nums, self.rules_panel = nums, rules
        self.wait(1.4)

    @beat("Play one game, honestly", seconds=23,
          narration="Play it once. The car is behind a door chosen at random; "
                    "you pick one; the host opens a goat door from what is "
                    "left. Now there are two doors closed — the one you picked "
                    "and one other. It looks like an even chance. It is not, "
                    "and the reason is that the host's choice was not free.")
    def one_game(self):
        runs = play_monty(seed=1, trials=TRIALS)
        g = next(r for r in runs if r.switch_wins)   # a game where switching wins
        self.play(FadeOut(self.rules_panel), run_time=0.3)

        pick_mark = Text("your pick", font_size=18, color=PICK_C).next_to(
            self.doors[g.first_pick], UP, buff=0.22)
        self.play(self.doors[g.first_pick].animate.set_stroke(PICK_C, width=4),
                  FadeIn(pick_mark), run_time=1.0)
        self.wait(0.6)

        goat = Text("goat", font_size=20, color=GOAT_C).move_to(
            self.doors[g.opened])
        self.play(self.doors[g.opened].animate.set_stroke(
                      GOAT_C, width=3).set_fill(GOAT_C, opacity=0.18),
                  FadeIn(goat), run_time=1.1)

        other = next(d for d in range(3) if d not in (g.first_pick, g.opened))
        stay_or = self.panel(
            VGroup(Text("stay on", font_size=20, color=DIM),
                   Text(f"door {g.first_pick + 1}", font_size=21, color=PICK_C)
                   ).arrange(RIGHT, buff=0.25),
            VGroup(Text("switch to", font_size=20, color=DIM),
                   Text(f"door {other + 1}", font_size=21, color=CAR_C)
                   ).arrange(RIGHT, buff=0.25),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in stay_or],
                              lag_ratio=0.35, run_time=1.4))
        self.game, self.goat, self.pick_mark = g, goat, pick_mark
        self.stay_or = stay_or
        self.wait(1.4)

    @beat("Why the host's constraint moves the odds", seconds=27,
          narration="Think about your first pick. One time in three it was the "
                    "car, and switching loses. Two times in three it was a "
                    "goat — and then the host, forbidden from opening the car "
                    "and forbidden from opening your door, has no choice at "
                    "all. He must open the other goat, which leaves the car "
                    "behind the door you switch to. Two cases out of three "
                    "where switching wins.")
    def argument(self):
        self.play(FadeOut(self.stay_or), FadeOut(self.goat),
                  FadeOut(self.pick_mark), run_time=0.5)
        self.play(*[d.animate.set_stroke(DOOR_C, width=2.6).set_fill(
                        BLACK, opacity=1.0) for d in self.doors],
                  run_time=0.5)

        cases = VGroup()
        for label, chance, verdict, col in (
                ("first pick was the car", "1/3", "switching loses", GOAT_C),
                ("first pick was a goat", "2/3", "switching wins", CAR_C)):
            cases.add(VGroup(
                Text(label, font_size=19, color=DIM),
                MathTex(chance, font_size=26, color=PICK_C),
                Text(verdict, font_size=19, color=col),
            ).arrange(DOWN, buff=0.16, aligned_edge=LEFT))
        cases.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        cases.move_to(RIGHT * 2.9 + UP * 0.9)

        self.play(FadeIn(cases[0], shift=LEFT * 0.2), run_time=1.0)
        self.wait(0.8)
        self.play(FadeIn(cases[1], shift=LEFT * 0.2), run_time=1.0)
        note = Text("the host has no choice here", font_size=18,
                    color=CAR_C).next_to(cases, DOWN, buff=0.45).align_to(
            cases, LEFT)
        self.play(FadeIn(note), run_time=0.8)
        self.cases = VGroup(cases, note)
        self.wait(1.6)

    @beat("Twenty thousand games settle it", seconds=19,
          narration="And if the argument still feels slippery, play it twenty "
                    "thousand times. Switching wins sixty-six point three "
                    "percent of them, staying wins thirty-three point seven, "
                    "and those add to one — which is the check that the host "
                    "was following his rule. Two thirds, counted rather than "
                    "reasoned.")
    def simulate(self):
        assert verify_monty_is_two_thirds()
        sw, st = monty_rates(play_monty(seed=1, trials=TRIALS))
        assert round(sw * 100, 1) == 66.3 and round(st * 100, 1) == 33.7
        assert abs(sw + st - 1.0) < 1e-9

        self.play(FadeOut(self.cases), run_time=0.4)

        bar_w = 5.2
        sw_bar = Rectangle(width=bar_w * sw, height=0.5, color=CAR_C,
                           fill_color=CAR_C, fill_opacity=0.65, stroke_width=2)
        st_bar = Rectangle(width=bar_w * st, height=0.5, color=GOAT_C,
                           fill_color=GOAT_C, fill_opacity=0.65, stroke_width=2)
        rows = VGroup(
            VGroup(Text("switch", font_size=20, color=CAR_C), sw_bar,
                   MathTex(rf"{sw:.1%}".replace("%", r"\%"), font_size=24, color=CAR_C)
                   ).arrange(RIGHT, buff=0.25),
            VGroup(Text("stay", font_size=20, color=GOAT_C), st_bar,
                   MathTex(rf"{st:.1%}".replace("%", r"\%"), font_size=24, color=GOAT_C)
                   ).arrange(RIGHT, buff=0.25),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        rows.move_to(DOWN * 1.9)

        head = Text(f"{TRIALS:,} games", font_size=21, color=DIM).next_to(
            rows, UP, buff=0.4).align_to(rows, LEFT)
        self.play(FadeIn(head), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows],
                              lag_ratio=0.4, run_time=1.8))

        exact = MathTex(r"\tfrac{2}{3}", font_size=40, color=CAR_C).move_to(
            RIGHT * 3.6 + UP * 1.2)
        self.play(Write(exact), run_time=1.0)
        self.wait(2.2)
