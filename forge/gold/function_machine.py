"""Gold scene 011 — a function as a machine, and as a curve.

Tier 1. Two pictures of the same object, and the connection between them is
what every later calculus scene assumes. A model that has only ever seen the
curve will not know the curve is a *record* of something.

Every output shown is computed by calling the function, so the machine's
answers and the plotted curve cannot drift apart.
"""

from manim import *

from forge.beats import ForgeScene, beat

IN_C = "#F0AC5F"
OUT_C = "#5CD0B3"
BOX_C = BLUE_C
DIM = GREY_B


def f(x: float) -> float:
    return 0.5 * x ** 2 - 1.0


class FunctionMachine(ForgeScene):

    samples = [-2.0, -1.0, 0.0, 1.0, 2.0]

    @beat("A machine that takes a number and returns one", seconds=8,
          narration="Think of a function as a machine. A number goes in, the "
                    "machine does something to it, and one number comes out. "
                    "The rule here is: square it, halve it, subtract one.")
    def machine(self):
        self.box = RoundedRectangle(width=2.6, height=1.5, corner_radius=0.18,
                                    color=BOX_C, stroke_width=3,
                                    fill_opacity=0.12, fill_color=BOX_C)
        self.box.move_to(LEFT * 3.3 + UP * 0.9)
        rule = MathTex(r"\tfrac{x^2}{2} - 1", font_size=34, color=BOX_C).move_to(self.box)
        self.in_arrow = Arrow(self.box.get_left() + LEFT * 1.1, self.box.get_left(),
                              buff=0.1, color=IN_C, stroke_width=4)
        self.out_arrow = Arrow(self.box.get_right(), self.box.get_right() + RIGHT * 1.1,
                               buff=0.1, color=OUT_C, stroke_width=4)
        self.title = Text("Two pictures of the same thing",
                          font_size=29).to_edge(UP, buff=0.4)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.box), Write(rule), run_time=1.3)
        self.play(GrowArrow(self.in_arrow), GrowArrow(self.out_arrow), run_time=0.9)
        self.wait(0.8)

    @beat("Feed it numbers and watch what comes out", seconds=10,
          narration="Put in minus two, and four comes back — no, one. Put in "
                    "zero and you get minus one. Each input produces exactly "
                    "one output. That is the whole requirement: never two "
                    "answers for the same question.")
    def feed(self):
        self.rows = VGroup()
        for x in self.samples:
            y = f(x)                    # computed, never typed
            lab_in = MathTex(f"{x:g}", font_size=30, color=IN_C)
            lab_in.next_to(self.in_arrow, LEFT, buff=0.15)
            lab_out = MathTex(f"{y:g}", font_size=30, color=OUT_C)
            lab_out.next_to(self.out_arrow, RIGHT, buff=0.15)
            self.play(FadeIn(lab_in, shift=RIGHT * 0.2), run_time=0.35)
            self.play(Indicate(self.box, color=BOX_C, scale_factor=1.06), run_time=0.35)
            self.play(FadeIn(lab_out, shift=RIGHT * 0.2), run_time=0.35)
            self.rows.add(VGroup(
                MathTex(rf"{x:g} \;\mapsto\; {y:g}", font_size=26, color=DIM)))
            self.play(FadeOut(lab_in), FadeOut(lab_out), run_time=0.25)

        self.rows.arrange(DOWN, buff=0.18).move_to([-3.3, -1.7, 0])
        self.play(LaggedStart(*[FadeIn(r) for r in self.rows],
                              lag_ratio=0.15, run_time=1.2))
        self.wait(0.7)

    @beat("Plot each pair as a point", seconds=9,
          narration="Now plot each pair. The input along the bottom, the output "
                    "up the side. Five inputs, five points. Nothing new has "
                    "happened — this is the same list, drawn instead of "
                    "written.")
    def plot(self):
        self.axes = Axes(x_range=[-3, 3, 1], y_range=[-2, 4, 1],
                         x_length=5.4, y_length=4.2,
                         axis_config={"include_tip": False, "stroke_width": 2,
                                      "color": GREY_B, "font_size": 18}
                         ).to_edge(RIGHT, buff=1.0).shift(DOWN * 0.2)
        self.play(Create(self.axes), run_time=1.2)

        dots = VGroup(*[Dot(self.axes.c2p(x, f(x)), color=OUT_C, radius=0.075)
                        for x in self.samples])
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots],
                              lag_ratio=0.2, run_time=1.6))
        self.dots = dots
        self.wait(0.8)

    @beat("Fill in every input between and a curve appears", seconds=9,
          narration="Feed it every number in between, not just five, and the "
                    "points merge into a curve. The curve is not a different "
                    "object from the machine. It is a record of every answer "
                    "the machine would ever give.")
    def curve(self):
        curve = self.axes.plot(f, x_range=[-3, 3], color=OUT_C, stroke_width=4)
        self.play(Create(curve), run_time=2.0)
        self.play(FadeOut(self.rows), run_time=0.5)

        caption = Text("every input, all at once", font_size=22, color=DIM)
        caption.next_to(self.axes, DOWN, buff=0.35)
        self.play(FadeIn(caption, shift=UP * 0.12), run_time=0.8)
        self.wait(1.6)
