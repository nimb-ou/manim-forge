"""Gold scene 031 — arithmetic on a clock face.

Number theory, Tier 2. Modular arithmetic is usually introduced as a rule
about remainders, which makes it look like a convention. On a clock it is
obviously not a convention: the numbers really do come back round, and the
question of which ones you land on has a visible answer.

The payoff is the cycle length. Stepping by five on a twelve-clock visits all
twelve; stepping by four visits three. That difference is exactly
m / gcd(m, step), which ``verify_cycle_is_m_over_gcd`` checks for every step
before the scene draws either walk -- so the rule is established on all twelve
cases, not inferred from the two the animation has room to show.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.numbers import (clock_positions, cycle_length, euclid,
                                      verify_cycle_is_m_over_gcd)

FACE_C = GREY_B
WALK_C = "#F0AC5F"
FULL_C = "#5CD0B3"
SHORT_C = "#FC6255"
DIM = GREY_B

M = 12
RADIUS = 2.2
FACE_AT = LEFT * 3.2 + DOWN * 0.3


class ModularClock(ForgeScene):

    def hour(self, k):
        a = PI / 2 - TAU * (k % M) / M
        return FACE_AT + RADIUS * np.array([np.cos(a), np.sin(a), 0.0])

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 3.3).align_to(UP * 2.3, UP)

    def draw_face(self):
        ring = Circle(radius=RADIUS, color=FACE_C, stroke_width=2.4).move_to(FACE_AT)
        marks, labels = VGroup(), {}
        for k in range(M):
            d = Dot(self.hour(k), color=FACE_C, radius=0.055)
            t = Text(str(k), font_size=17, color=DIM).move_to(
                FACE_AT + (RADIUS + 0.34) * (self.hour(k) - FACE_AT) / RADIUS)
            labels[k] = (d, t)
            marks.add(VGroup(d, t))
        return ring, marks, labels

    @beat("Numbers that come back round", seconds=18,
          narration="A clock face with twelve positions. Counting past eleven "
                    "does not give twelve — it gives zero, because the "
                    "numbers come back round. This is not a convention anyone "
                    "agreed to. It is what a circle does, and writing it down "
                    "as remainders came afterwards.")
    def face(self):
        self.title = Text("Arithmetic on a clock face",
                          font_size=30).to_edge(UP, buff=0.4)
        ring, marks, self.labels = self.draw_face()
        self.ring, self.marks = ring, marks

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(ring), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(m, scale=0.6) for m in marks],
                              lag_ratio=0.07, run_time=1.6))

        rule = self.panel(
            MathTex(r"11 + 1 \equiv 0", font_size=32, color=WALK_C),
            MathTex(r"\pmod{12}", font_size=24, color=DIM),
        )
        self.play(Write(rule[0]), run_time=1.0)
        self.play(FadeIn(rule[1]), run_time=0.5)
        self.rule = rule
        self.wait(1.4)

    @beat("Step by five, and visit everything", seconds=17,
          narration="Start at zero and keep adding five. Five, ten, then "
                    "three, eight, one — the walk jumps around the face "
                    "seemingly at random, but look at what it covers. Every "
                    "single position, exactly once, before it returns to "
                    "zero. Twelve steps, twelve positions.")
    def step_five(self):
        assert verify_cycle_is_m_over_gcd(M)
        step = 5
        pos = clock_positions(M, step, cycle_length(M, step))
        assert pos[:6] == [0, 5, 10, 3, 8, 1]     # spoken in the narration
        assert len(set(pos[:-1])) == M

        self.play(FadeOut(self.rule), run_time=0.3)
        head = self.panel(
            VGroup(Text("step by", font_size=20, color=DIM),
                   MathTex(rf"{step}", font_size=30, color=WALK_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN))

        self.play(FadeIn(head), run_time=0.6)
        lines, visited = VGroup(), set()
        for a, b in zip(pos, pos[1:]):
            ln = Line(self.hour(a), self.hour(b), color=WALK_C,
                      stroke_width=2.2, stroke_opacity=0.75)
            lines.add(ln)
            visited.add(b)
            self.play(Create(ln),
                      self.labels[b][0].animate.set_color(FULL_C).scale(1.4),
                      run_time=0.34)

        count = VGroup(
            MathTex(rf"{len(set(pos[:-1]))}", font_size=34, color=FULL_C),
            Text("of 12 positions", font_size=19, color=DIM),
        ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN)
        count.next_to(head, DOWN, buff=0.5).align_to(head, LEFT)
        self.play(FadeIn(count, shift=UP * 0.12), run_time=0.8)
        self.five = VGroup(lines, head, count)
        self.wait(1.4)

    @beat("Step by four, and most of the face is unreachable", seconds=20,
          narration="Now step by four instead. Four, eight, zero — and it is "
                    "already back where it started, having touched three "
                    "positions out of twelve. Nine of them can never be "
                    "reached this way, no matter how long you keep adding "
                    "four. Same clock, same kind of step, completely "
                    "different reach.")
    def step_four(self):
        step = 4
        pos = clock_positions(M, step, cycle_length(M, step))
        assert pos == [0, 4, 8, 0]
        assert len(set(pos[:-1])) == 3
        assert M - 3 == 9                        # spoken

        self.play(FadeOut(self.five), run_time=0.5)
        self.play(*[self.labels[k][0].animate.set_color(FACE_C).scale(1 / 1.4)
                    for k in range(M)], run_time=0.5)

        head = self.panel(
            VGroup(Text("step by", font_size=20, color=DIM),
                   MathTex(rf"{step}", font_size=30, color=SHORT_C)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN))
        self.play(FadeIn(head), run_time=0.6)

        lines = VGroup()
        for a, b in zip(pos, pos[1:]):
            ln = Line(self.hour(a), self.hour(b), color=SHORT_C, stroke_width=2.6)
            lines.add(ln)
            self.play(Create(ln),
                      self.labels[b][0].animate.set_color(SHORT_C).scale(1.4),
                      run_time=0.5)

        unreachable = [k for k in range(M) if k not in set(pos)]
        self.play(LaggedStart(*[self.labels[k][1].animate.set_color("#3A3A3A")
                                for k in unreachable],
                              lag_ratio=0.08, run_time=1.2))
        count = VGroup(
            VGroup(MathTex("3", font_size=30, color=SHORT_C),
                   Text("reachable", font_size=19, color=DIM)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(MathTex(rf"{len(unreachable)}", font_size=30, color="#3A3A3A"),
                   Text("never", font_size=19, color=DIM)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
        ).arrange(DOWN, buff=0.26, aligned_edge=LEFT)
        count.next_to(head, DOWN, buff=0.5).align_to(head, LEFT)
        self.play(LaggedStart(*[FadeIn(c, shift=LEFT * 0.2) for c in count],
                              lag_ratio=0.35, run_time=1.2))
        self.four = VGroup(lines, head, count)
        self.wait(1.4)

    @beat("The difference is a common factor", seconds=27,
          narration="What separates them is whether the step shares a factor "
                    "with twelve. Five shares nothing with twelve, so it "
                    "cannot settle into a short loop and has to visit "
                    "everything. Four and twelve share a factor of four, and "
                    "the walk closes after twelve divided by four — three "
                    "steps. The reach is twelve over the greatest common "
                    "divisor, and that holds for every step, not just these "
                    "two.")
    def why(self):
        import math as _m
        assert _m.gcd(12, 5) == 1 and _m.gcd(12, 4) == 4
        assert cycle_length(M, 5) == 12 and cycle_length(M, 4) == 3
        assert verify_cycle_is_m_over_gcd(M)

        self.play(FadeOut(self.four), run_time=0.4)
        rows = VGroup()
        for s in range(1, M):
            g = _m.gcd(M, s)
            rows.add(VGroup(
                MathTex(rf"{s}", font_size=19, color=DIM),
                MathTex(rf"\gcd={g}", font_size=17, color=DIM),
                MathTex(rf"{cycle_length(M, s)}", font_size=21,
                        color=FULL_C if g == 1 else SHORT_C),
            ).arrange(RIGHT, buff=0.18))
        rows.arrange_in_grid(rows=6, cols=2, buff=(0.5, 0.18))
        rows.move_to(RIGHT * 3.4 + UP * 0.9)

        law = MathTex(r"\frac{12}{\gcd(12,\,\text{step})}", font_size=34,
                      color=FULL_C).next_to(rows, DOWN, buff=0.55)

        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.15) for r in rows],
                              lag_ratio=0.1, run_time=2.0))
        self.play(Write(law), run_time=1.2)
        self.wait(2.2)
