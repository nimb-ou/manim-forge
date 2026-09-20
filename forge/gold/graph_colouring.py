"""Gold scene 020 — the fewest colours that work.

Graphs, Tier 3. The scene exists to show a greedy algorithm being *wrong* --
not crashing, not obviously failing, but returning a valid answer that is one
colour worse than necessary, on a graph where nothing looks unusual.

The graph is a crown: three nodes on the left, three on the right, each joined
to the two opposite it but not the one across from it. Taken left-then-right
it takes two colours. Taken alternately it takes three. Same graph, same
algorithm, different order. ``chromatic_number`` settles which is right by
exhaustive search, so "one more than necessary" is a checked statement rather
than a claim about a picture.

That is the honest version of "NP-hard": not that the problem is impossible,
but that the cheap method can be beaten and knowing when costs you a search.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.graphs import (Graph, chromatic_number,
                                     greedy_colouring, verify_colouring)

PALETTE = ["#5CD0B3", "#F0AC5F", "#FC6255", "#8B7FD4"]
NODE_C = GREY_B
BAD_C = "#FC6255"
DIM = GREY_B

LAYOUT = LEFT * 2.0 + DOWN * 0.2
SC = 1.25

#: A crown graph: u_i joins every v_j except the one opposite it.
G = Graph(
    nodes={**{f"u{i}": (-1.5, 1.5 - i * 1.5) for i in range(3)},
           **{f"v{i}": (1.5, 1.5 - i * 1.5) for i in range(3)}},
    edges=[(f"u{i}", f"v{j}", 1.0)
           for i in range(3) for j in range(3) if i != j],
)

SIDE_FIRST = ["u0", "u1", "u2", "v0", "v1", "v2"]
ALTERNATING = ["u0", "v0", "u1", "v1", "u2", "v2"]


class GraphColouring(ForgeScene):

    def pos(self, n):
        x, y = G.nodes[n]
        return LAYOUT + np.array([x * SC, y * SC, 0.0])

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.2).align_to(UP * 2.4, UP)

    def build_nodes(self):
        self.circles, self.labels = {}, {}
        group = VGroup()
        for n in G.nodes:
            c = Circle(radius=0.28, color=NODE_C, stroke_width=2.6,
                       fill_color=BLACK, fill_opacity=1.0).move_to(self.pos(n))
            t = Text(n, font_size=18, color=NODE_C).move_to(c)
            self.circles[n], self.labels[n] = c, t
            group.add(VGroup(c, t))
        return group

    def reset_colours(self):
        return [c.animate.set_fill(BLACK, opacity=1.0).set_stroke(NODE_C, width=2.6)
                for c in self.circles.values()] + \
               [t.animate.set_color(NODE_C) for t in self.labels.values()]

    def apply_order(self, order, run_time=0.55):
        """Colour the graph in ``order``, one node at a time, on screen."""
        colours = {}
        for n in order:
            used = {colours[m] for m, _ in G.neighbours(n) if m in colours}
            c = 0
            while c in used:
                c += 1
            colours[n] = c
            self.play(self.circles[n].animate.set_fill(
                          PALETTE[c], opacity=0.85).set_stroke(PALETTE[c], width=3.4),
                      self.labels[n].animate.set_color(BLACK),
                      run_time=run_time)
        # The on-screen result must equal what the primitive computes; if the
        # inline rule above ever drifts, the scene would narrate one thing and
        # show another.
        assert colours == greedy_colouring(G, order)
        assert verify_colouring(G, colours)
        return colours

    @beat("Colour the nodes so no edge joins two the same", seconds=20,
          narration="A network, and one rule: give every node a colour so "
                    "that no edge ever joins two nodes of the same colour. "
                    "This is timetabling in disguise — the nodes are exams, "
                    "the edges are students sitting both, and the colours are "
                    "time slots. Fewer colours means a shorter exam period.")
    def rule(self):
        self.title = Text("The fewest colours that work",
                          font_size=30).to_edge(UP, buff=0.4)

        self.edge_lines = {}
        lines = VGroup()
        for a, b, _ in G.edges:
            ln = Line(self.pos(a), self.pos(b), color=GREY_E, stroke_width=2.2)
            self.edge_lines[(a, b)] = ln
            lines.add(ln)
        nodes = self.build_nodes()

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(LaggedStart(*[Create(l) for l in lines],
                              lag_ratio=0.07, run_time=1.8))
        self.play(LaggedStart(*[FadeIn(n, scale=0.6) for n in nodes],
                              lag_ratio=0.1, run_time=1.4))

        rule = self.panel(
            Text("rule", font_size=20, color=DIM),
            Text("joined nodes differ", font_size=23, color=PALETTE[0]),
            Text("goal", font_size=20, color=DIM),
            Text("use as few as possible", font_size=23, color=PALETTE[1]),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rule],
                              lag_ratio=0.3, run_time=2.0))
        self.rule_panel = rule
        self.wait(1.2)

    @beat("The obvious method: take the lowest colour that fits", seconds=20,
          narration="The obvious method is to go through the nodes one at a "
                    "time and give each the lowest-numbered colour none of "
                    "its neighbours has yet. It is fast, it never breaks the "
                    "rule, and here it finishes in two colours. Left side "
                    "green, right side amber, and every edge crosses between "
                    "them.")
    def greedy_good(self):
        self.play(FadeOut(self.rule_panel), run_time=0.4)

        label = Text("order: left side, then right", font_size=21,
                     color=DIM).next_to(self.title, DOWN, buff=0.32)
        self.play(FadeIn(label), run_time=0.6)

        colours = self.apply_order(SIDE_FIRST)
        used = len(set(colours.values()))

        result = self.panel(
            VGroup(MathTex(rf"{used}", font_size=40, color=PALETTE[0]),
                   Text("colours", font_size=21, color=DIM)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            Text("valid", font_size=20, color=PALETTE[0]),
        )
        self.play(FadeIn(result, shift=LEFT * 0.2), run_time=0.9)
        self.first_label, self.first_result = label, result
        self.first_used = used
        self.wait(1.4)

    @beat("Same graph, same method, a different order", seconds=22,
          narration="Now change nothing except the order. Take one node from "
                    "the left, then one from the right, and alternate. The "
                    "rule is applied exactly as before and the answer is "
                    "still valid — but it uses three colours instead of two. "
                    "The graph did not change. The algorithm did not change. "
                    "Only the order did.")
    def greedy_bad(self):
        self.play(FadeOut(self.first_result), *self.reset_colours(),
                  run_time=0.8)
        new_label = Text("order: alternating sides", font_size=21,
                         color=BAD_C).next_to(self.title, DOWN, buff=0.32)
        self.play(FadeTransform(self.first_label, new_label), run_time=0.6)

        colours = self.apply_order(ALTERNATING)
        used = len(set(colours.values()))
        assert used > self.first_used        # the point of the whole scene

        result = self.panel(
            VGroup(MathTex(rf"{used}", font_size=40, color=BAD_C),
                   Text("colours", font_size=21, color=DIM)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            Text("still valid", font_size=20, color=DIM),
            Text("still worse", font_size=22, color=BAD_C),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in result],
                              lag_ratio=0.3, run_time=1.6))
        self.second_label, self.second_result = new_label, result
        self.second_used = used
        self.wait(1.6)

    @beat("Knowing the true minimum means searching for it", seconds=26,
          narration="So which is right? Two is, and finding that out took a "
                    "search through every way of colouring the graph — "
                    "hundreds of assignments for six nodes, and rising as a "
                    "power of the number of nodes. That is the real shape of "
                    "the difficulty. A valid colouring is easy and a good one "
                    "is usually easy. Being certain you cannot do better is "
                    "what costs.")
    def hardness(self):
        chi = chromatic_number(G)
        assert chi == self.first_used
        assert chi < self.second_used

        self.play(FadeOut(self.second_result), FadeOut(self.second_label),
                  run_time=0.5)
        self.play(*self.reset_colours(), run_time=0.6)
        best = self.apply_order(SIDE_FIRST, run_time=0.32)
        assert len(set(best.values())) == chi

        n = len(G.nodes)
        rows = VGroup(
            VGroup(Text("greedy, good order", font_size=19, color=DIM),
                   MathTex(rf"{self.first_used}", font_size=28, color=PALETTE[0])
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
            VGroup(Text("greedy, bad order", font_size=19, color=DIM),
                   MathTex(rf"{self.second_used}", font_size=28, color=BAD_C)
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
            VGroup(Text("true minimum", font_size=19, color=DIM),
                   MathTex(rf"{chi}", font_size=32, color=PALETTE[0])
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
        ).arrange(DOWN, buff=0.28, aligned_edge=LEFT)

        cost = VGroup(
            Text("to be sure, check", font_size=19, color=DIM),
            MathTex(rf"{chi}^{{{n}}} = {chi ** n}", font_size=28, color=DIM),
            Text("assignments", font_size=19, color=DIM),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)

        panel = self.panel(rows, cost)
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.3, run_time=1.8))
        self.play(FadeIn(cost, shift=UP * 0.12), run_time=1.0)
        self.wait(2.4)
