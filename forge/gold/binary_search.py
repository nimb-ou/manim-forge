"""Gold scene 025 — halving the haystack.

Discrete, Tier 2. The scene exists to make "logarithmic" mean something
physical: not a curve on a graph, but the number of times you can halve
sixty-four before nothing is left.

The probes come from ``binary_search`` itself, so the midpoints on screen are
the midpoints the code picks -- floor division and inclusive bounds included.
A scene that halves to a prettier index is animating a different algorithm
from the one it names, and nothing in the picture would betray it.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.algorithms import (binary_search, linear_search_cost,
                                         verify_binary_search)

CELL_C = GREY_C
LIVE_C = BLUE_C
PROBE_C = "#F0AC5F"
GONE_C = "#3A3A3A"
FOUND_C = "#5CD0B3"
DIM = GREY_B

N = 64
TARGET = 41
ROWS, COLS = 4, 16
CELL, GAP = 0.44, 0.06
GRID_AT = LEFT * 1.6 + DOWN * 0.2


class BinarySearch(ForgeScene):

    def cell_pos(self, i):
        r, c = divmod(i, COLS)
        w = COLS * (CELL + GAP)
        h = ROWS * (CELL + GAP)
        return GRID_AT + np.array([c * (CELL + GAP) - w / 2 + CELL / 2,
                                   h / 2 - r * (CELL + GAP) - CELL / 2, 0.0])

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.6).align_to(UP * 2.3, UP)

    @beat("Sixty-four numbers, sorted, and one to find", seconds=16,
          narration="Sixty-four numbers, in order, and one of them is the one "
                    "we want. Checked one at a time from the left, the number "
                    "forty-one is the forty-first thing you look at. Being "
                    "sorted ought to be worth more than that.")
    def grid(self):
        self.title = Text("Halving the haystack", font_size=30).to_edge(UP, buff=0.4)
        self.items = list(range(1, N + 1))

        self.boxes, self.labels = [], []
        group = VGroup()
        for i, v in enumerate(self.items):
            sq = Square(side_length=CELL, color=CELL_C, stroke_width=1.6,
                        fill_color=BLACK, fill_opacity=1.0).move_to(self.cell_pos(i))
            t = Text(str(v), font_size=13, color=DIM).move_to(sq)
            self.boxes.append(sq)
            self.labels.append(t)
            group.add(VGroup(sq, t))

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(g, scale=0.7) for g in group],
                              lag_ratio=0.008, run_time=2.0))

        want = self.panel(
            VGroup(Text("find", font_size=20, color=DIM),
                   MathTex(rf"{TARGET}", font_size=34, color=FOUND_C)
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
            VGroup(Text("one at a time", font_size=19, color=DIM),
                   MathTex(rf"{linear_search_cost(self.items, TARGET)}",
                           font_size=28, color="#FC6255")
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
            Text("looks", font_size=17, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(w, shift=LEFT * 0.2) for w in want],
                              lag_ratio=0.3, run_time=1.5))
        self.want = want
        self.wait(1.2)

    @beat("Look in the middle, and throw half away", seconds=24,
          narration="Instead, look at the middle one. It is thirty-two, which "
                    "is too small — so every number to its left is too small "
                    "as well, and half the list is gone in a single look. "
                    "Look at the middle of what remains. Too large this time, "
                    "so the upper half goes. Each look costs one comparison "
                    "and buys half the remaining list.")
    def probe(self):
        assert verify_binary_search(self.items)
        self.probes = binary_search(self.items, TARGET)
        assert self.probes[0].value == 32        # spoken in the narration
        assert self.probes[0].verdict == "too low"

        self.play(FadeOut(self.want), run_time=0.3)
        self.trace = VGroup()
        self.trace.move_to(RIGHT * 4.6 + UP * 2.0)

        rows = VGroup()
        for k, p in enumerate(self.probes):
            span = p.hi - p.lo + 1
            box = self.boxes[p.mid]
            self.play(box.animate.set_stroke(PROBE_C, width=3.5).set_fill(
                          PROBE_C, opacity=0.35),
                      self.labels[p.mid].animate.set_color(WHITE),
                      run_time=0.5)

            if p.verdict == "found":
                break
            dead = (range(p.lo, p.mid) if p.verdict == "too high"
                    else range(p.mid + 1, p.hi + 1))
            self.play(*[self.boxes[i].animate.set_stroke(GONE_C, width=1.0)
                        for i in dead],
                      *[self.labels[i].animate.set_color(GONE_C) for i in dead],
                      self.boxes[p.mid].animate.set_stroke(GONE_C, width=1.0).set_fill(
                          BLACK, opacity=1.0),
                      self.labels[p.mid].animate.set_color(GONE_C),
                      run_time=0.55)

            row = VGroup(
                MathTex(rf"{p.value}", font_size=22, color=PROBE_C),
                Text(p.verdict, font_size=16, color=DIM),
                MathTex(rf"{span} \to {span - len(list(dead)) - 1}",
                        font_size=20, color=LIVE_C),
            ).arrange(RIGHT, buff=0.2)
            rows.add(row)
            rows.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
            rows.move_to(RIGHT * 4.6 + UP * 1.6)
            self.play(FadeIn(row, shift=LEFT * 0.15), run_time=0.3)
        self.rows = rows
        self.wait(0.8)

    @beat("Found, and the count is the answer", seconds=15,
          narration="Forty-one, in six looks rather than forty-one. And six is "
                    "not a coincidence: sixty-four halved six times is one. "
                    "The number of looks is the number of halvings, which is "
                    "the logarithm base two of the list length.")
    def found(self):
        last = self.probes[-1]
        assert last.verdict == "found" and last.value == TARGET
        n_looks = len(self.probes)
        assert n_looks == 6                      # spoken in the narration
        assert 2 ** n_looks == N                 # "halved six times is one"

        box = self.boxes[last.mid]
        self.play(box.animate.set_stroke(FOUND_C, width=4).set_fill(
                      FOUND_C, opacity=0.6),
                  self.labels[last.mid].animate.set_color(BLACK),
                  run_time=0.8)
        self.play(Flash(box, color=FOUND_C, line_length=0.2,
                        flash_radius=0.4, num_lines=12), run_time=0.9)

        self.play(FadeOut(self.rows), run_time=0.4)
        halving = VGroup(*[
            MathTex(rf"{N // 2 ** k}", font_size=24,
                    color=LIVE_C if k < n_looks else FOUND_C)
            for k in range(n_looks + 1)]).arrange(RIGHT, buff=0.3)
        arrows = VGroup(*[
            MathTex(r"\to", font_size=18, color=DIM) for _ in range(n_looks)])
        chain = VGroup()
        for i, m in enumerate(halving):
            chain.add(m)
            if i < n_looks:
                chain.add(arrows[i])
        chain.arrange(RIGHT, buff=0.16).move_to(RIGHT * 4.3 + UP * 1.4).scale(0.85)

        self.play(LaggedStart(*[FadeIn(c) for c in chain],
                              lag_ratio=0.12, run_time=1.8))
        self.chain = chain
        self.wait(1.4)

    @beat("Why this scales and counting does not", seconds=20,
          narration="The difference matters most when the list is large. "
                    "Double the numbers and one-at-a-time searching doubles "
                    "its work, while halving needs exactly one more look. A "
                    "million entries take twenty looks. A billion take thirty. "
                    "That is what logarithmic means, and it is why sorted data "
                    "is worth the trouble of sorting.")
    def scaling(self):
        import math as _m
        sizes = [64, 1_000_000, 1_000_000_000]
        rows = VGroup()
        for s in sizes:
            looks = _m.ceil(_m.log2(s))
            rows.add(VGroup(
                Text(f"{s:,}", font_size=20, color=DIM),
                MathTex(r"\to", font_size=18, color=DIM),
                MathTex(rf"{looks}", font_size=28, color=FOUND_C),
                Text("looks", font_size=17, color=DIM),
            ).arrange(RIGHT, buff=0.2))
        rows.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        rows.move_to(RIGHT * 4.4 + DOWN * 0.3)
        # The narration says twenty and thirty; both are ceil(log2).
        assert _m.ceil(_m.log2(1_000_000)) == 20
        assert _m.ceil(_m.log2(1_000_000_000)) == 30

        self.play(FadeOut(self.chain), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.35, run_time=1.8))
        law = MathTex(r"\log_2 n", font_size=36, color=FOUND_C).next_to(
            rows, DOWN, buff=0.6)
        self.play(Write(law), run_time=1.0)
        self.wait(2.4)
