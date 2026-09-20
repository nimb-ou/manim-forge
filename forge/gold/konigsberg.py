"""Gold scene 019 — the walk that cannot be done.

Graphs, Tier 3. Konigsberg is the scene where an animation can do something a
proof on paper cannot: fail, visibly, several times, before explaining why
failure was guaranteed. Watching three attempts strand themselves is what
makes the counting argument land as an explanation rather than an assertion.

Nothing here is staged. The bridges are a real multigraph -- four of the seven
are doubled pairs, and ``degree`` counts multiplicities, because two bridges
between the same banks are two ways across. The parity argument is run by the
primitive: four land masses are odd, a walk has two ends, so no such walk
exists. ``verify_konigsberg_has_no_walk`` asserts it before the scene says it.

The attempted routes are checked too: each is a genuine walk along real
bridges that genuinely runs out, not a path drawn to look stuck.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.graphs import (KONIGSBERG, euler_path_exists,
                                     odd_degree_nodes, verify_walk_strands,
                                     verify_konigsberg_has_no_walk,
                                     walk_until_stuck)

LAND_C = "#8D7350"
WATER_C = "#2A4A63"
BRIDGE_C = GREY_B
WALK_C = "#5CD0B3"
STUCK_C = "#FC6255"
ODD_C = "#F0AC5F"
DIM = GREY_B

LAYOUT = LEFT * 2.6
SC = 1.15

#: Which of the seven bridges runs where. Parallel bridges are bowed apart by
#: ``bend`` so both are visible; they are the same connection either way.
BRIDGES = [("N", "A", -0.45), ("N", "A", 0.45),
           ("S", "A", -0.45), ("S", "A", 0.45),
           ("N", "B", 0.0), ("S", "B", 0.0), ("A", "B", 0.0)]

NAMES = {"N": "north bank", "A": "the island", "B": "east quarter",
         "S": "south bank"}

# walk_until_stuck returns indices into KONIGSBERG.edges, and the scene
# looks them up in BRIDGES. The two lists must stay in lockstep.
assert [(a, b) for a, b, _ in BRIDGES] == \
       [(a, b) for a, b, _ in KONIGSBERG.edges]


class Konigsberg(ForgeScene):

    def pos(self, n):
        x, y = KONIGSBERG.nodes[n]
        return LAYOUT + np.array([x * SC, y * SC, 0.0])

    def bridge(self, a, b, bend):
        start, end = self.pos(a), self.pos(b)
        if abs(bend) < 1e-9:
            return Line(start, end, color=BRIDGE_C, stroke_width=4)
        return ArcBetweenPoints(start, end, angle=bend * 2,
                                color=BRIDGE_C, stroke_width=4)

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 3.9).align_to(UP * 2.4, UP)

    @beat("Four pieces of land, seven bridges", seconds=17,
          narration="Konigsberg, seventeen thirty-six. A river splits the "
                    "city into four pieces: two banks, an island, and a "
                    "quarter to the east. Seven bridges join them. The "
                    "townspeople asked a question that sounds like a puzzle "
                    "and turned out to start a branch of mathematics.")
    def city(self):
        self.title = Text("The seven bridges of Konigsberg",
                          font_size=30).to_edge(UP, buff=0.4)

        self.lands, self.land_labels = {}, {}
        blobs = VGroup()
        for n in KONIGSBERG.nodes:
            c = Circle(radius=0.42, color=LAND_C, fill_color=LAND_C,
                       fill_opacity=0.45, stroke_width=3).move_to(self.pos(n))
            t = Text(n, font_size=24, color=WHITE).move_to(c)
            self.lands[n], self.land_labels[n] = c, t
            blobs.add(VGroup(c, t))

        self.spans = []
        arcs = VGroup()
        for a, b, bend in BRIDGES:
            arc = self.bridge(a, b, bend)
            self.spans.append(((a, b), arc))
            arcs.add(arc)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(b, scale=0.6) for b in blobs],
                              lag_ratio=0.15, run_time=1.4))
        self.play(LaggedStart(*[Create(a) for a in arcs],
                              lag_ratio=0.16, run_time=2.4))

        legend = self.panel(*[
            VGroup(Text(k, font_size=22, color=LAND_C),
                   Text(NAMES[k], font_size=19, color=DIM)
                   ).arrange(RIGHT, buff=0.3)
            for k in ("N", "A", "B", "S")])
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in legend],
                              lag_ratio=0.2, run_time=1.4))
        self.legend = legend
        self.wait(1.0)

    @beat("The question, and three honest attempts", seconds=27,
          narration="Can you walk through the city crossing every bridge "
                    "exactly once? Try it. Start on the north bank and you "
                    "strand yourself with a bridge still unused. Start on the "
                    "island instead — stranded again. Start in the east "
                    "quarter and you barely get going. Something is always "
                    "left over, and no amount of trying tells you whether you "
                    "are unlucky or whether it cannot be done at all.")
    def attempts(self):
        question = Text("cross every bridge exactly once?",
                        font_size=24, color=WALK_C).next_to(
            self.title, DOWN, buff=0.35)
        self.play(FadeIn(question), run_time=0.8)
        self.play(FadeOut(self.legend), run_time=0.4)

        # The routes are walked, not written. Three attempts drafted by eye
        # turned out to be one illegal walk and two that could have carried
        # on: the narration said "stranded" and the bridges disagreed. Every
        # maximal walk here strands, because no Euler path exists -- so the
        # plain edge order is enough, and each one is checked before it plays.
        self.tally = None
        for start_node in ("N", "A", "B"):
            walk = walk_until_stuck(KONIGSBERG, start_node)
            assert verify_walk_strands(KONIGSBERG, walk, start_node)

            used = VGroup()
            marker = Dot(self.pos(start_node), color=WALK_C, radius=0.11)
            self.play(FadeIn(marker, scale=0.5), run_time=0.35)
            for step in walk:
                arc = self.spans[step.edge_index][1]
                self.play(arc.animate.set_stroke(WALK_C, width=6),
                          marker.animate.move_to(self.pos(step.to)),
                          run_time=0.4)
                used.add(arc)

            left = len(BRIDGES) - len(walk)
            msg = Text(f"{left} bridge{'s' if left != 1 else ''} unused",
                       font_size=22, color=STUCK_C).next_to(
                question, DOWN, buff=0.45)
            if self.tally is None:
                self.play(FadeIn(msg), run_time=0.5)
            else:
                self.play(FadeTransform(self.tally, msg), run_time=0.5)
            self.tally = msg
            self.play(Flash(marker, color=STUCK_C, line_length=0.2,
                            flash_radius=0.34, num_lines=10), run_time=0.6)
            self.play(FadeOut(marker),
                      *[a.animate.set_stroke(BRIDGE_C, width=4) for a in used],
                      run_time=0.5)
        self.question = question
        self.wait(0.8)

    @beat("Throw the map away and keep the connections", seconds=26,
          narration="Euler's move was to stop looking at the map. The shape "
                    "of the island does not matter, nor the length of any "
                    "bridge, nor which way the river bends. The only thing "
                    "that decides the answer is which pieces of land are "
                    "joined, and by how many bridges. Shrink each piece of "
                    "land to a dot and the whole problem is four dots and "
                    "seven lines.")
    def abstract(self):
        self.play(FadeOut(self.question), FadeOut(self.tally), run_time=0.5)
        self.play(*[c.animate.set_fill(opacity=0.0).scale(0.62)
                    for c in self.lands.values()],
                  *[t.animate.scale(0.82) for t in self.land_labels.values()],
                  run_time=1.6)

        counts = {n: KONIGSBERG.degree(n) for n in KONIGSBERG.nodes}
        assert sum(counts.values()) == 2 * len(BRIDGES)   # handshake lemma

        rows = VGroup(*[
            VGroup(Text(n, font_size=22, color=LAND_C),
                   Text("touches", font_size=18, color=DIM),
                   MathTex(rf"{counts[n]}", font_size=28, color=ODD_C),
                   Text("bridges", font_size=18, color=DIM),
                   ).arrange(RIGHT, buff=0.22)
            for n in ("N", "A", "B", "S")])
        rows.arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        total = VGroup(
            MathTex(rf"{sum(counts.values())}", font_size=26, color=DIM),
            Text("ends, for 7 bridges", font_size=18, color=DIM),
        ).arrange(RIGHT, buff=0.22)
        self.degrees = self.panel(rows, total)

        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.28, run_time=1.8))
        self.play(FadeIn(total), run_time=0.7)
        self.wait(1.2)

    @beat("Count the odd ones, and the answer is forced", seconds=29,
          narration="Here is the argument. Any piece of land you pass through "
                    "costs two bridges — one to arrive, one to leave. So a "
                    "land touched by an odd number of bridges can only be "
                    "where you start or where you finish. A walk has two "
                    "ends. But every one of these four is odd: five, three, "
                    "three, three. Four candidates for two positions. The "
                    "walk does not exist, and no cleverness will find it.")
    def parity(self):
        assert verify_konigsberg_has_no_walk()
        odd = odd_degree_nodes(KONIGSBERG)
        assert not euler_path_exists(KONIGSBERG)
        assert len(odd) == 4
        # the narration says "five, three, three, three"
        assert sorted(KONIGSBERG.degree(n) for n in KONIGSBERG.nodes) \
            == [3, 3, 3, 5]

        self.play(FadeOut(self.degrees), run_time=0.4)
        self.play(LaggedStart(*[
            self.lands[n].animate.set_stroke(ODD_C, width=5) for n in odd],
            lag_ratio=0.2, run_time=1.4))

        rule = self.panel(
            Text("pass through a land", font_size=20, color=DIM),
            Text("= 2 bridges", font_size=22, color=WALK_C),
            Text("odd land = an end", font_size=20, color=ODD_C),
            Text("a walk has 2 ends", font_size=20, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rule],
                              lag_ratio=0.35, run_time=2.2))
        self.wait(1.4)

        verdict = VGroup(
            VGroup(MathTex(rf"{len(odd)}", font_size=40, color=ODD_C),
                   Text("odd lands", font_size=21, color=DIM)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            Text("more than 2", font_size=22, color=STUCK_C),
            Text("no such walk exists", font_size=26, color=STUCK_C),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        verdict.next_to(rule, DOWN, buff=0.6).align_to(rule, LEFT)

        self.play(FadeIn(verdict[0], shift=UP * 0.12), run_time=0.8)
        self.play(FadeIn(verdict[1]), run_time=0.6)
        self.play(Write(verdict[2]), run_time=1.2)
        self.wait(2.4)
