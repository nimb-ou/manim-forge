"""Gold scene 010 — area survives rearrangement.

Tier 1, and load-bearing. The Pythagorean proof, the circle-area derivation
and Euclid's algorithm all rest on one idea: cutting a shape and moving the
pieces does not change how much area there is. A model that has seen that
established can be taught the proofs; one that has not will draw them as
decoration.

Areas are computed from the shapes' own dimensions and asserted equal before
the scene claims they are.
"""

from manim import *

from forge.beats import ForgeScene, beat

SHAPE_C = BLUE_C
CUT_C = "#F0AC5F"
RESULT_C = "#5CD0B3"
DIM = GREY_B

UNIT = 0.62          # world units per grid square
BASE, HEIGHT, SKEW = 6, 3, 2      # parallelogram, in grid squares


class AreaRearranged(ForgeScene):

    @beat("A parallelogram, and the awkward question of its area", seconds=13,
          narration="A parallelogram. How much area is inside it? The formula "
                    "is easy enough to look up, but where does it come from? "
                    "It leans, so we cannot simply count rows of squares.")
    def setup(self):
        w, h, s = BASE * UNIT, HEIGHT * UNIT, SKEW * UNIT
        self.origin = LEFT * 3.4 + DOWN * 1.3

        self.para = Polygon(
            self.origin, self.origin + RIGHT * w,
            self.origin + RIGHT * (w + s) + UP * h, self.origin + RIGHT * s + UP * h,
            color=SHAPE_C, stroke_width=3, fill_opacity=0.28, fill_color=SHAPE_C)
        self.title = Text("Cutting and moving changes nothing",
                          font_size=29).to_edge(UP, buff=0.4)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.para), run_time=1.5)
        self.wait(1.0)

    @beat("Cut the leaning triangle off one end", seconds=14,
          narration="So cut. Drop a vertical line from the top-left corner, and "
                    "the parallelogram splits into a triangle and a "
                    "quadrilateral. Nothing has been added and nothing removed "
                    "— we have only drawn a line.")
    def cut(self):
        w, h, s = BASE * UNIT, HEIGHT * UNIT, SKEW * UNIT
        o = self.origin

        self.tri = Polygon(o + RIGHT * s + UP * h, o + RIGHT * s, o,
                           color=CUT_C, stroke_width=2.5,
                           fill_opacity=0.45, fill_color=CUT_C)
        self.rest = Polygon(o + RIGHT * s, o + RIGHT * w,
                            o + RIGHT * (w + s) + UP * h, o + RIGHT * s + UP * h,
                            color=SHAPE_C, stroke_width=2.5,
                            fill_opacity=0.28, fill_color=SHAPE_C)
        knife = DashedLine(o + RIGHT * s + UP * h, o + RIGHT * s,
                           color=WHITE, stroke_width=2.5, dash_length=0.1)

        self.play(Create(knife), run_time=0.9)
        # Sweep by colour rather than by reference: animations leave stage
        # copies the tracked reference no longer points at, and a surviving
        # outline of the original shape makes the rearrangement unreadable.
        self.remove(*[m for m in self.mobjects
                      if isinstance(m, Polygon) and m is not self.para])
        self.remove(self.para)
        self.add(self.rest, self.tri)
        self.play(FadeOut(knife), run_time=0.3)
        self.play(self.tri.animate.set_fill(opacity=0.6), run_time=0.5)
        self.wait(0.7)

    @beat("Slide it round to the other side", seconds=12,
          narration="Now slide that triangle across to the far end. It fits "
                    "exactly, because the two slanted edges were parallel and "
                    "the same length. And what is left is a rectangle.")
    def slide(self):
        w = BASE * UNIT
        self.play(self.tri.animate.shift(RIGHT * w), run_time=1.8,
                  rate_func=rate_functions.ease_in_out_sine)
        self.play(self.tri.animate.set_color(RESULT_C).set_fill(RESULT_C, opacity=0.35),
                  self.rest.animate.set_color(RESULT_C).set_fill(RESULT_C, opacity=0.35),
                  run_time=0.9)
        self.wait(0.9)

    @beat("A rectangle we can measure by counting", seconds=18,
          narration="And a rectangle we can measure by counting: six squares "
                    "across, three up, eighteen in total. That number belonged "
                    "to the parallelogram all along — base times height, not "
                    "because of a formula, but because the parallelogram is a "
                    "rectangle that has been pushed over.")
    def count(self):
        w, h = BASE * UNIT, HEIGHT * UNIT
        o = self.origin + RIGHT * (SKEW * UNIT)

        squares = VGroup()
        for r in range(HEIGHT):
            for c in range(BASE):
                squares.add(Square(side_length=UNIT, stroke_width=1,
                                   stroke_color=WHITE, fill_opacity=0)
                            .move_to(o + RIGHT * (c + 0.5) * UNIT + UP * (r + 0.5) * UNIT))
        self.play(LaggedStart(*[Create(s) for s in squares],
                              lag_ratio=0.02, run_time=1.8))

        # Computed from the shape's own dimensions, and checked.
        area = BASE * HEIGHT
        assert area == len(squares)

        panel = VGroup(
            Text(f"{BASE} x {HEIGHT}", font_size=32, color=DIM),
            Text(str(area), font_size=76, color=RESULT_C),
            Text("squares of area", font_size=21, color=DIM),
        ).arrange(DOWN, buff=0.16).to_edge(RIGHT, buff=0.9).shift(UP * 0.4)
        self.play(FadeIn(panel[0]), run_time=0.5)
        self.play(FadeIn(panel[1], scale=0.7), Write(panel[2]), run_time=1.0)

        rule = MathTex(r"A = b \times h", font_size=40, color=RESULT_C)
        rule.next_to(panel, DOWN, buff=0.55)
        self.play(Write(rule), run_time=1.0)
        self.wait(1.5)
