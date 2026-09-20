"""Gold scene 018 — Dijkstra, and why being greedy is safe here.

Graphs, Tier 2, building on the breadth-first scene. The hook is chosen so the
two disagree: the direct edge from A to D is one hop and costs fourteen, while
the cheapest route is three hops and costs nine. Fewest steps and lowest cost
are different questions, and a scene that does not show them diverging has not
explained why a second algorithm exists.

Node D is the whole algorithm in one place: its estimate falls fourteen, then
eleven, then nine as better routes are found, and only then is it settled. The
animation reads that sequence from ``dijkstra_steps`` rather than staging it,
including the relaxations that improved nothing -- those are what make the
greedy choice look reckless and still be correct.

The final distances are checked against exhaustive enumeration of every simple
path before the scene calls them cheapest.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.graphs import (Graph, dijkstra, dijkstra_steps, path_to,
                                     verify_dijkstra_matches_bruteforce)

NODE_C = GREY_B
DONE_C = BLUE_C
LIVE_C = "#F0AC5F"
PATH_C = "#5CD0B3"
STALE_C = "#FC6255"
DIM = GREY_B

LAYOUT = LEFT * 1.5 + DOWN * 0.25
SC = 1.15

G = Graph(
    nodes={"A": (-2.7, 0.2), "B": (-1.0, 1.9), "C": (1.0, 1.9),
           "D": (2.7, 0.2), "E": (1.0, -1.8), "F": (-1.0, -1.8)},
    edges=[("A", "B", 4), ("B", "C", 3), ("C", "D", 2), ("D", "E", 6),
           ("E", "F", 3), ("F", "A", 2), ("A", "D", 14), ("B", "E", 9)],
)


class Dijkstra(ForgeScene):

    def pos(self, n):
        x, y = G.nodes[n]
        return LAYOUT + np.array([x * SC, y * SC, 0.0])

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.7).align_to(UP * 2.4, UP)

    def edge_key(self, a, b):
        return (a, b) if (a, b) in self.edges else (b, a)

    @beat("The same network, but the edges cost different amounts", seconds=19,
          narration="The same kind of network, with one change: crossing an "
                    "edge now costs something, and the costs differ. Counting "
                    "steps no longer answers the question. Look at A and D. "
                    "There is a single edge between them — one step, the "
                    "fewest possible — and taking it costs fourteen.")
    def weighted(self):
        self.title = Text("Cheapest route, not fewest steps",
                          font_size=30).to_edge(UP, buff=0.4)

        self.edges, self.weights = {}, {}
        lines, wlabels = VGroup(), VGroup()
        for a, b, w in G.edges:
            ln = Line(self.pos(a), self.pos(b), color=GREY_E, stroke_width=2.6)
            lab = MathTex(rf"{w:.0f}", font_size=22, color=DIM).move_to(
                ln.point_from_proportion(0.5)
                + normalize(rotate_vector(ln.get_unit_vector(), PI / 2)) * 0.26)
            self.edges[(a, b)] = ln
            self.weights[(a, b)] = lab
            lines.add(ln)
            wlabels.add(lab)

        self.circles, self.labels = {}, {}
        dots = VGroup()
        for n in G.nodes:
            c = Circle(radius=0.27, color=NODE_C, stroke_width=2.4,
                       fill_color=BLACK, fill_opacity=1.0).move_to(self.pos(n))
            t = Text(n, font_size=21, color=NODE_C).move_to(c)
            self.circles[n], self.labels[n] = c, t
            dots.add(VGroup(c, t))

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(LaggedStart(*[Create(l) for l in lines],
                              lag_ratio=0.07, run_time=1.6))
        self.play(LaggedStart(*[FadeIn(d, scale=0.6) for d in dots],
                              lag_ratio=0.09, run_time=1.3))
        self.play(LaggedStart(*[FadeIn(w) for w in wlabels],
                              lag_ratio=0.07, run_time=1.2))

        direct = self.edges[("A", "D")]
        self.play(direct.animate.set_stroke(STALE_C, width=4.5),
                  self.weights[("A", "D")].animate.set_color(STALE_C),
                  run_time=1.0)
        self.wait(1.0)
        self.play(direct.animate.set_stroke(GREY_E, width=2.6),
                  self.weights[("A", "D")].animate.set_color(DIM),
                  run_time=0.6)

    @beat("Keep a running estimate, and improve it", seconds=24,
          narration="Give every node a running estimate of the cheapest way "
                    "to reach it: zero for A, unknown for the rest. Then take "
                    "the cheapest node not yet finished, and look at what it "
                    "leads to. If going through it beats a node's current "
                    "estimate, lower the estimate. Watch D. It starts at "
                    "fourteen, drops to eleven, and finally to nine.")
    def relax(self):
        self.steps = dijkstra_steps(G, "A")

        self.est = {}
        for n in G.nodes:
            txt = "0" if n == "A" else r"\infty"
            col = DONE_C if n == "A" else DIM
            m = MathTex(txt, font_size=24, color=col).next_to(
                self.circles[n], UP, buff=0.14)
            self.est[n] = m
        self.play(LaggedStart(*[FadeIn(m, scale=0.6) for m in self.est.values()],
                              lag_ratio=0.1, run_time=1.4))

        for s in self.steps:
            self.play(self.circles[s.node].animate.set_stroke(LIVE_C, width=4.5),
                      self.labels[s.node].animate.set_color(LIVE_C),
                      run_time=0.45)
            for r in s.relaxations:
                key = self.edge_key(r.frm, r.to)
                colour = PATH_C if r.improved else STALE_C
                self.play(self.edges[key].animate.set_stroke(colour, width=4),
                          run_time=0.3)
                if r.improved:
                    new = MathTex(rf"{r.new:.0f}", font_size=24,
                                  color=PATH_C).move_to(self.est[r.to])
                    self.play(FadeTransform(self.est[r.to], new), run_time=0.4)
                    self.est[r.to] = new
                else:
                    self.play(Indicate(self.est[r.to], color=STALE_C,
                                       scale_factor=1.15), run_time=0.4)
                self.play(self.edges[key].animate.set_stroke(GREY_E, width=2.6),
                          run_time=0.25)
            self.play(self.circles[s.node].animate.set_stroke(DONE_C, width=3.4),
                      self.labels[s.node].animate.set_color(DONE_C),
                      self.est[s.node].animate.set_color(DONE_C),
                      run_time=0.4)

    @beat("Why settling the cheapest one is safe", seconds=29,
          narration="The greedy step looks reckless: once a node is settled "
                    "its estimate is never revisited. It is safe because of "
                    "the order. Any other route to that node has to leave "
                    "through some node that is still unsettled, and every one "
                    "of those already costs at least as much — so the "
                    "detour cannot come back cheaper. That argument needs "
                    "every cost to be positive, and it fails the moment one "
                    "is not.")
    def why_greedy(self):
        order = [s.node for s in self.steps]
        costs = [s.dist for s in self.steps]

        rows = VGroup(*[
            VGroup(MathTex(rf"{i+1}.", font_size=22, color=DIM),
                   Text(n, font_size=22, color=DONE_C),
                   MathTex(rf"{c:.0f}", font_size=26, color=PATH_C),
                   ).arrange(RIGHT, buff=0.28)
            for i, (n, c) in enumerate(zip(order, costs))])
        rows.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        head = Text("settled in order", font_size=20, color=DIM)
        panel = self.panel(head, rows)

        self.play(FadeIn(head), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.22, run_time=2.0))
        # Non-decreasing settle order is the property the argument rests on.
        assert costs == sorted(costs)
        note = Text("never decreases", font_size=19, color=PATH_C).next_to(
            rows, DOWN, buff=0.4).align_to(rows, LEFT)
        self.play(FadeIn(note, shift=UP * 0.12), run_time=0.8)
        self.panel_g = VGroup(panel, note)
        self.wait(1.6)

    @beat("The answer, checked against every route", seconds=22,
          narration="The cheapest way from A to D costs nine, and it takes "
                    "three steps rather than one. Before saying that, the "
                    "scene enumerates every route through this network and "
                    "confirms there is nothing cheaper. Greedy algorithms are "
                    "exactly the ones that look right while being wrong on a "
                    "case the drawing happens not to contain.")
    def answer(self):
        assert verify_dijkstra_matches_bruteforce(G, "A")
        dist, prev = dijkstra(G, "A")
        route = path_to(prev, "D")

        self.play(FadeOut(self.panel_g), run_time=0.5)

        anims = []
        for a, b in zip(route, route[1:]):
            anims.append(self.edges[self.edge_key(a, b)].animate.set_stroke(
                PATH_C, width=5.5))
        for n in route:
            anims.append(self.circles[n].animate.set_stroke(PATH_C, width=4.5))
        self.play(LaggedStart(*anims, lag_ratio=0.15, run_time=1.8))

        panel = self.panel(
            MathTex(r"\;\to\;".join(route), font_size=28, color=PATH_C),
            VGroup(Text("cost", font_size=20, color=DIM),
                   MathTex(rf"{dist['D']:.0f}", font_size=32, color=PATH_C)
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
            VGroup(Text("direct edge", font_size=20, color=DIM),
                   MathTex(r"14", font_size=28, color=STALE_C)
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
            Text("checked against every route", font_size=17, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(p, shift=LEFT * 0.2) for p in panel],
                              lag_ratio=0.3, run_time=1.8))
        self.wait(2.2)
