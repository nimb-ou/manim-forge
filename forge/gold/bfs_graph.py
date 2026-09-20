"""Gold scene 017 — breadth-first search, expanding in waves.

Graphs, Tier 2, and the first graph scene in the corpus. The visual grammar it
establishes -- nodes as labelled circles, edges behind them, a frontier that
advances a ring at a time -- is what every later graph scene inherits.

The traversal is not choreographed. ``bfs`` returns the real sequence of
visits with the depth and the frontier at each one, and the animation reads
that sequence. If the algorithm changed, the animation would change with it,
which is the only way the picture can be trusted to be of the algorithm.

The closing claim -- that BFS gives shortest paths when every edge costs one
-- is checked against exhaustive search before it is spoken.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.graphs import (Graph, bfs, path_to,
                                     verify_bfs_is_shortest_unweighted)

NODE_C = GREY_B
SEEN_C = BLUE_C
FRONT_C = "#F0AC5F"
PATH_C = "#5CD0B3"
DIM = GREY_B

LAYOUT = LEFT * 1.6 + DOWN * 0.3
SC = 1.25

G = Graph(
    nodes={"A": (-2.4, 0.0), "B": (-1.1, 1.3), "C": (-1.1, -1.3),
           "D": (0.2, 2.0), "E": (0.2, 0.4), "F": (0.2, -2.0),
           "G": (1.5, 1.3), "H": (1.5, -0.9), "I": (2.8, 0.2)},
    edges=[("A", "B", 1), ("A", "C", 1), ("B", "D", 1), ("B", "E", 1),
           ("C", "E", 1), ("C", "F", 1), ("D", "G", 1), ("E", "G", 1),
           ("E", "H", 1), ("F", "H", 1), ("G", "I", 1), ("H", "I", 1)],
)


class BreadthFirst(ForgeScene):

    def pos(self, n):
        x, y = G.nodes[n]
        return LAYOUT + np.array([x * SC, y * SC, 0.0])

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.32, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.6).align_to(UP * 2.3, UP)

    @beat("A graph, and a place to start", seconds=17,
          narration="A network: things, and which of them are connected. We "
                    "start at A and want to reach everything else. The only "
                    "question is what order to look in — and the order turns "
                    "out to decide what the search is good for.")
    def draw_graph(self):
        self.title = Text("Breadth-first search", font_size=30).to_edge(UP, buff=0.4)

        self.edge_lines = {}
        lines = VGroup()
        for a, b, _ in G.edges:
            ln = Line(self.pos(a), self.pos(b), color=GREY_E, stroke_width=2.4)
            self.edge_lines[(a, b)] = ln
            lines.add(ln)

        self.circles, self.labels = {}, {}
        dots = VGroup()
        for n in G.nodes:
            c = Circle(radius=0.26, color=NODE_C, stroke_width=2.4,
                       fill_color=BLACK, fill_opacity=1.0).move_to(self.pos(n))
            t = Text(n, font_size=21, color=NODE_C).move_to(c)
            self.circles[n], self.labels[n] = c, t
            dots.add(VGroup(c, t))

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(LaggedStart(*[Create(l) for l in lines],
                              lag_ratio=0.06, run_time=1.6))
        self.play(LaggedStart(*[FadeIn(d, scale=0.6) for d in dots],
                              lag_ratio=0.08, run_time=1.4))

        self.play(self.circles["A"].animate.set_stroke(SEEN_C, width=4),
                  self.labels["A"].animate.set_color(SEEN_C), run_time=0.7)
        start = Text("start", font_size=18, color=SEEN_C).next_to(
            self.circles["A"], LEFT, buff=0.22)
        self.play(FadeIn(start), run_time=0.5)
        self.start_lab = start
        self.wait(0.8)

    @beat("Everything one step away, then everything two", seconds=19,
          narration="Breadth-first means exactly what it says: finish the "
                    "whole of one ring before starting the next. Everything "
                    "one step from A. Then everything two steps away. Then "
                    "three. The search grows as a wave rather than a thread, "
                    "and no node is ever reached before its turn.")
    def waves(self):
        steps = bfs(G, "A")
        by_depth = {}
        for s in steps:
            by_depth.setdefault(s.depth, []).append(s)

        self.depth_of = {s.node: s.depth for s in steps}
        self.via = {s.node: s.via for s in steps}

        ring_lab = None
        for depth in sorted(by_depth):
            if depth == 0:
                continue
            group = by_depth[depth]
            new_lab = Text(f"{depth} step{'s' if depth > 1 else ''} away",
                           font_size=24, color=FRONT_C).next_to(
                self.title, DOWN, buff=0.35)
            if ring_lab is None:
                self.play(FadeIn(new_lab), run_time=0.5)
            else:
                self.play(FadeTransform(ring_lab, new_lab), run_time=0.5)
            ring_lab = new_lab

            anims = []
            for s in group:
                key = (s.via, s.node) if (s.via, s.node) in self.edge_lines \
                    else (s.node, s.via)
                anims.append(self.edge_lines[key].animate.set_stroke(
                    FRONT_C, width=3.6))
                anims.append(self.circles[s.node].animate.set_stroke(
                    FRONT_C, width=4))
                anims.append(self.labels[s.node].animate.set_color(FRONT_C))
            self.play(LaggedStart(*anims, lag_ratio=0.1, run_time=1.5))
            self.wait(0.45)
            # Settle the ring to "seen" so the next frontier is the only thing
            # highlighted -- a frontier that never dims stops being a frontier.
            self.play(*[self.circles[s.node].animate.set_stroke(SEEN_C, width=3)
                        for s in group],
                      *[self.labels[s.node].animate.set_color(SEEN_C)
                        for s in group],
                      run_time=0.5)
        self.ring_lab = ring_lab
        self.wait(0.8)

    @beat("The depth it was found at is the distance", seconds=19,
          narration="Here is what the wave bought us. Because rings are "
                    "finished in order, the ring a node was found in is the "
                    "fewest steps from the start to it. Not an estimate — the "
                    "answer. When every edge costs the same, breadth-first "
                    "search solves the shortest-path problem for free.")
    def distances(self):
        assert verify_bfs_is_shortest_unweighted(G, "A")

        self.play(FadeOut(self.ring_lab), FadeOut(self.start_lab), run_time=0.4)

        tags = VGroup()
        for n, d in self.depth_of.items():
            t = MathTex(rf"{d}", font_size=22, color=PATH_C).next_to(
                self.circles[n], UR, buff=0.06)
            tags.add(t)
        self.play(LaggedStart(*[FadeIn(t, scale=0.6) for t in tags],
                              lag_ratio=0.12, run_time=1.8))
        self.tags = tags

        note = self.panel(
            Text("ring found in", font_size=20, color=DIM),
            Text("= steps from A", font_size=22, color=PATH_C),
        )
        self.play(FadeIn(note, shift=LEFT * 0.2), run_time=0.9)
        self.note = note
        self.wait(1.8)

    @beat("Follow the links back to get the route", seconds=17,
          narration="And the route itself is already recorded. Each node "
                    "remembers which neighbour reached it first, so walking "
                    "those links backwards from any node returns the shortest "
                    "path to it. From I back to A: four steps, and there is "
                    "no shorter way.")
    def trace_back(self):
        prev = {n: self.via[n] for n in self.depth_of}
        route = path_to(prev, "I")
        assert route[0] == "A" and route[-1] == "I"
        assert len(route) - 1 == self.depth_of["I"]

        self.play(FadeOut(self.tags), FadeOut(self.note), run_time=0.4)

        anims = []
        for a, b in zip(route, route[1:]):
            key = (a, b) if (a, b) in self.edge_lines else (b, a)
            anims.append(self.edge_lines[key].animate.set_stroke(PATH_C, width=5.5))
        for n in route:
            anims.append(self.circles[n].animate.set_stroke(PATH_C, width=4.5))
            anims.append(self.labels[n].animate.set_color(PATH_C))
        self.play(LaggedStart(*anims, lag_ratio=0.12, run_time=2.2))

        panel = self.panel(
            Text("A to I", font_size=22, color=DIM),
            MathTex(r"\;\to\;".join(route), font_size=26, color=PATH_C),
            VGroup(MathTex(rf"{len(route)-1}", font_size=30, color=PATH_C),
                   Text("steps, and no fewer", font_size=19, color=DIM)
                   ).arrange(RIGHT, buff=0.22, aligned_edge=DOWN),
        )
        self.play(LaggedStart(*[FadeIn(p, shift=LEFT * 0.2) for p in panel],
                              lag_ratio=0.35, run_time=1.6))
        self.wait(2.0)
