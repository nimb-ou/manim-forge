"""Gold scene 002 — learning rate and gradient descent.

The whole scene is driven by ``forge.primitives.optimise.Descent``. Nothing
about any trajectory is authored: the ball goes where the update rule sends it,
the divergence diverges because the arithmetic diverges, and the reported final
positions are read off the run rather than typed in.

That is the property worth training on. A model that learns to draw a plausible
descent curve has learned decoration. A model that learns to *run the optimiser
and animate the result* has learned something true.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.optimise import Descent, minimum_of

CURVE_C = BLUE_C
BALL_C = YELLOW_C
GOOD_C = "#5CD0B3"
BAD_C = "#FC6255"
SLOW_C = "#F0AC5F"

X_LO, X_HI = -5.0, 5.0


def loss(x: float) -> float:
    return 0.35 * x ** 2 + 0.4 * x + 1.2


class LearningRate(ForgeScene):

    x0 = 3.4

    # -- helpers -------------------------------------------------------------

    def make_axes(self) -> Axes:
        return Axes(
            x_range=[X_LO, X_HI, 1], y_range=[0, 11, 2],
            x_length=7.4, y_length=4.9,
            axis_config={"include_tip": False, "stroke_width": 2,
                         "color": GREY_B, "font_size": 20},
        ).to_edge(LEFT, buff=0.7).shift(DOWN * 0.35)

    def on_curve(self, x: float):
        return self.axes.c2p(x, loss(x))

    def run_descent(self, lr: float, n: int = 14) -> Descent:
        return Descent(loss, x0=self.x0, lr=lr, n_steps=n)

    def panel_text(self, *lines, color=WHITE, size=26) -> VGroup:
        return VGroup(*[Text(l, font_size=size, color=color) for l in lines]) \
            .arrange(DOWN, buff=0.16, aligned_edge=LEFT).move_to([4.2, 1.4, 0])

    # -- beats ---------------------------------------------------------------

    @beat("Draw the loss curve and mark its true minimum", seconds=5,
          narration="Here is a loss curve. Every point along it is one setting of our parameter, and the height is how badly the model does there. Somewhere down in that valley is the best setting.")
    def setup_curve(self):
        self.axes = self.make_axes()
        self.curve = self.axes.plot(loss, x_range=[X_LO, X_HI], color=CURVE_C,
                                    stroke_width=4)
        self.title = Text("Gradient descent", font_size=30).to_edge(UP, buff=0.35)

        min_x, min_y = minimum_of(loss, X_LO, X_HI)     # found, not assumed
        self.min_dot = Dot(self.axes.c2p(min_x, min_y), color=GOOD_C, radius=0.07)
        # Below the x-axis, not between dot and axis: the curve's minimum sits
        # close enough to y=0 that a label placed under the dot lands on the
        # tick marks.
        self.min_label = Text("minimum", font_size=20, color=GOOD_C)
        self.min_label.next_to(self.axes.c2p(min_x, 0), DOWN, buff=0.28)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), Create(self.axes), run_time=1.2)
        self.play(Create(self.curve), run_time=1.6)
        self.play(FadeIn(self.min_dot, scale=0.5), Write(self.min_label), run_time=0.8)
        self.wait(0.3)

    @beat("Place a ball on the slope and show the gradient", seconds=5,
          narration="Start anywhere. The slope under your feet tells you which way is downhill — and how steep it is tells you how far to step.")
    def show_gradient(self):
        d = self.run_descent(lr=0.55)
        g = d.steps[0].grad

        self.ball = Dot(self.on_curve(self.x0), color=BALL_C, radius=0.11)
        span = 1.15
        tangent = Line(
            self.axes.c2p(self.x0 - span, loss(self.x0) - g * span),
            self.axes.c2p(self.x0 + span, loss(self.x0) + g * span),
            color=BALL_C, stroke_width=3,
        )

        self.rule = self.panel_text("the slope tells you", "which way is downhill",
                                    color=GREY_B, size=23)
        self.play(FadeIn(self.ball, scale=0.4), run_time=0.6)
        self.play(Create(tangent), FadeIn(self.rule, shift=UP * 0.15), run_time=1.2)
        self.wait(0.6)
        self.play(FadeOut(tangent), FadeOut(self.rule), run_time=0.5)

    @beat("A good learning rate walks the ball to the minimum", seconds=8,
          narration="Step a fraction of that slope, again and again. That fraction is the learning rate. Get it right and you walk straight into the valley.")
    def good_rate(self):
        d = self.run_descent(lr=0.55)
        label = self.panel_text("learning rate", "0.55", color=GOOD_C, size=30)
        self.play(FadeIn(label, shift=UP * 0.15), run_time=0.6)

        trail = VGroup()
        for s in d.steps[:10]:
            mark = Dot(self.on_curve(s.x), color=GOOD_C, radius=0.045).set_opacity(0.55)
            trail.add(mark)
            self.play(
                self.ball.animate.move_to(self.on_curve(s.next_x)),
                FadeIn(mark), run_time=0.28, rate_func=rate_functions.ease_in_out_sine,
            )

        verdict = Text(f"settles at x = {d.final_x:.2f}", font_size=24, color=GOOD_C)
        verdict.next_to(label, DOWN, buff=0.5)
        self.play(Write(verdict), run_time=0.7)
        self.wait(0.7)
        self.play(FadeOut(trail), FadeOut(label), FadeOut(verdict), run_time=0.6)

    @beat("Too large a step and it climbs out of the valley", seconds=8,
          narration="But make that fraction too big and each step overshoots the bottom, landing further up the other side. The steps grow, and you climb out of the valley entirely.")
    def too_large(self):
        d = self.run_descent(lr=3.5)
        label = self.panel_text("learning rate", "3.5", color=BAD_C, size=30)

        self.play(self.ball.animate.move_to(self.on_curve(self.x0)),
                  FadeIn(label, shift=UP * 0.15), run_time=0.7)

        # Only the steps still inside the frame can be drawn; the rest is the point.
        visible = [s for s in d.steps if X_LO < s.next_x < X_HI][:6]
        for s in visible:
            self.play(self.ball.animate.move_to(self.on_curve(s.next_x)),
                      run_time=0.32, rate_func=rate_functions.ease_in_out_sine)

        arrow = Arrow(self.axes.c2p(X_HI - 1.4, 8.2), self.axes.c2p(X_HI, 10.4),
                      color=BAD_C, buff=0, stroke_width=5)
        gone = Text("off the chart", font_size=22, color=BAD_C).next_to(arrow, LEFT, buff=0.12)
        self.play(self.ball.animate.move_to(self.axes.c2p(X_HI - 0.2, 10.0)).set_opacity(0.3),
                  GrowArrow(arrow), FadeIn(gone), run_time=0.9)

        verdict = Text(f"reaches x = {d.final_x:,.0f}", font_size=24, color=BAD_C)
        verdict.next_to(label, DOWN, buff=0.5)
        self.play(Write(verdict), run_time=0.7)
        self.wait(0.7)
        self.play(FadeOut(arrow), FadeOut(gone), FadeOut(label), FadeOut(verdict),
                  self.ball.animate.set_opacity(1.0), run_time=0.6)

    @beat("Too small and it never arrives", seconds=7,
          narration="Too small, and every step is timid. You are heading the right way, but you will run out of patience long before you arrive.")
    def too_small(self):
        d = self.run_descent(lr=0.15)
        label = self.panel_text("learning rate", "0.15", color=SLOW_C, size=30)
        self.play(self.ball.animate.move_to(self.on_curve(self.x0)),
                  FadeIn(label, shift=UP * 0.15), run_time=0.7)

        trail = VGroup()
        for s in d.steps:
            mark = Dot(self.on_curve(s.x), color=SLOW_C, radius=0.04).set_opacity(0.5)
            trail.add(mark)
            self.play(self.ball.animate.move_to(self.on_curve(s.next_x)),
                      FadeIn(mark), run_time=0.2)

        verdict = Text(f"only x = {d.final_x:.2f} after {len(d.steps)} steps",
                       font_size=22, color=SLOW_C)
        verdict.next_to(label, DOWN, buff=0.5)
        self.play(Write(verdict), run_time=0.7)
        self.wait(0.8)
        self.play(FadeOut(trail), FadeOut(label), FadeOut(verdict), run_time=0.6)

    @beat("The three rates side by side", seconds=6,
          narration="Too small crawls. Too large diverges. The whole art is finding the step that is just big enough to make progress, and just small enough to stay in the valley.")
    def compare(self):
        rows = VGroup()
        for lr, colour, note in ((0.15, SLOW_C, "too small"),
                                 (0.55, GOOD_C, "about right"),
                                 (3.5, BAD_C, "too large")):
            d = self.run_descent(lr)
            rows.add(VGroup(
                Text(f"{lr}", font_size=26, color=colour),
                Text(note, font_size=20, color=GREY_B),
            ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN))
        rows.arrange(DOWN, buff=0.34, aligned_edge=LEFT).move_to([4.2, 0.6, 0])

        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows],
                              lag_ratio=0.25, run_time=1.6))
        self.play(Indicate(rows[1], color=GOOD_C, scale_factor=1.12), run_time=0.9)
        self.wait(1.0)
