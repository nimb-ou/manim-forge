"""Gold scene 037 — sliding one thing over another.

Machine learning, Tier 2. Convolution is a sliding weighted sum, and every
useful thing about it follows from what the weights are. The scene uses one
signal and three kernels: a blur that sums to one, an edge detector that sums
to zero, and a sharpener. Same operation, three completely different jobs,
decided entirely by three numbers.

The outputs are computed by ``convolve1d``. Edges hold the boundary value
rather than padding with zeros -- zero padding makes the ends dip toward zero,
and a scene showing that dip would be presenting an artefact of the padding as
an effect of the filter.

``verify_blur_preserves_total`` checks that a kernel summing to one leaves the
signal's mean alone, which is the property that separates a blur from an
accidental gain.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.learning import convolve1d, verify_blur_preserves_total

SIG_C = BLUE_C
OUT_C = "#5CD0B3"
KERN_C = "#F0AC5F"
EDGE_C = "#FC6255"
DIM = GREY_B

SIGNAL = [1.0, 1.0, 1.2, 1.0, 3.0, 3.2, 3.0, 3.1, 1.0, 1.1, 0.9, 1.0,
          2.0, 2.1, 2.0, 1.0]
BLUR = [0.25, 0.5, 0.25]
EDGE = [-1.0, 0.0, 1.0]
SHARP = [-0.5, 2.0, -0.5]

BAR_W = 0.42
BAR_GAP = 0.06


class Convolution(ForgeScene):

    def bars(self, values, base_y, colour, scale=0.55, lo=None):
        """A row of bars, one per sample, growing up from ``base_y``."""
        g = VGroup()
        n = len(values)
        total = n * (BAR_W + BAR_GAP)
        for i, v in enumerate(values):
            h = max(abs(v) * scale, 0.02)
            r = Rectangle(width=BAR_W, height=h, color=colour,
                          fill_color=colour, fill_opacity=0.72,
                          stroke_width=1.2)
            x = -total / 2 + i * (BAR_W + BAR_GAP) + BAR_W / 2 - 0.6
            y = base_y + (h / 2 if v >= 0 else -h / 2)
            r.move_to(np.array([x, y, 0.0]))
            g.add(r)
        return g

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 4.9).align_to(UP * 2.3, UP)

    @beat("A signal, and a small window of weights", seconds=18,
          narration="A signal: sixteen numbers in a row, with two raised "
                    "patches and some jitter. Below it, a window three wide "
                    "holding three weights. Slide the window along, and at "
                    "each position multiply what is underneath by the weights "
                    "and add. That is the whole operation.")
    def signal(self):
        self.title = Text("Sliding one thing over another",
                          font_size=30).to_edge(UP, buff=0.4)
        self.sig_bars = self.bars(SIGNAL, 0.9, SIG_C)
        self.base = Line(np.array([-5.4, 0.9, 0.0]), np.array([2.5, 0.9, 0.0]),
                         color=GREY_D, stroke_width=1.6)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.base), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in self.sig_bars],
                              lag_ratio=0.05, run_time=1.8))

        self.window = Rectangle(width=3 * (BAR_W + BAR_GAP), height=2.0,
                                color=KERN_C, stroke_width=2.6)
        self.window.move_to(self.sig_bars[1].get_center()
                            + np.array([0.0, 0.55, 0.0]))
        self.play(Create(self.window), run_time=0.8)

        note = self.panel(
            Text("multiply and add", font_size=20, color=KERN_C),
            Text("at every position", font_size=20, color=DIM),
        )
        self.play(LaggedStart(*[FadeIn(n, shift=LEFT * 0.2) for n in note],
                              lag_ratio=0.3, run_time=1.2))
        self.note = note
        self.wait(1.2)

    @beat("Weights that sum to one: a blur", seconds=27,
          narration="Give the window a quarter, a half, a quarter. Every "
                    "output is a gentle average of a value and its two "
                    "neighbours, so the jitter flattens while the two patches "
                    "survive. The weights add to one, which is why the signal "
                    "does not get brighter or dimmer overall — checked, "
                    "because a kernel that quietly sums to more than one looks "
                    "like a blur and is also a gain.")
    def blur(self):
        assert verify_blur_preserves_total(SIGNAL, BLUR)
        out = convolve1d(SIGNAL, BLUR)

        self.play(FadeOut(self.note), run_time=0.3)
        kern = self.panel(
            VGroup(*[MathTex(rf"{w}", font_size=24, color=KERN_C) for w in BLUR]
                   ).arrange(RIGHT, buff=0.3),
            VGroup(Text("sums to", font_size=18, color=DIM),
                   MathTex(rf"{sum(BLUR):.0f}", font_size=24, color=KERN_C)
                   ).arrange(RIGHT, buff=0.22),
        )
        self.play(FadeIn(kern, shift=LEFT * 0.2), run_time=1.0)

        # Slide the window across before the output appears, so the output is
        # seen to be produced rather than simply displayed.
        for i in (4, 8, 12, 14):
            self.play(self.window.animate.move_to(
                self.sig_bars[i].get_center() + np.array([0.0, 0.55, 0.0])),
                run_time=0.5)

        out_bars = self.bars(out, -1.9, OUT_C)
        out_base = Line(np.array([-5.4, -1.9, 0.0]), np.array([2.5, -1.9, 0.0]),
                        color=GREY_D, stroke_width=1.6)
        self.play(Create(out_base), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in out_bars],
                              lag_ratio=0.06, run_time=1.8))

        mean_in = sum(SIGNAL) / len(SIGNAL)
        mean_out = sum(out) / len(out)
        rows = VGroup(
            VGroup(Text("mean in", font_size=18, color=DIM),
                   MathTex(rf"{mean_in:.3f}", font_size=22, color=SIG_C)
                   ).arrange(RIGHT, buff=0.22),
            VGroup(Text("mean out", font_size=18, color=DIM),
                   MathTex(rf"{mean_out:.3f}", font_size=22, color=OUT_C)
                   ).arrange(RIGHT, buff=0.22),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        rows.next_to(kern, DOWN, buff=0.55).align_to(kern, LEFT)
        self.play(FadeIn(rows, shift=UP * 0.1), run_time=0.9)
        self.out_bars, self.out_base = out_bars, out_base
        self.kern, self.rows = kern, rows
        self.wait(1.4)

    @beat("Weights that sum to zero: an edge detector", seconds=23,
          narration="Now change nothing but the three numbers. Minus one, "
                    "zero, plus one. Where the signal is flat, the left and "
                    "right neighbours cancel and the output is nothing at all. "
                    "Where the signal steps up or down, they do not cancel and "
                    "the output spikes. The same sliding sum, and it has "
                    "stopped smoothing and started finding edges.")
    def edges(self):
        out = convolve1d(SIGNAL, EDGE)
        assert abs(sum(EDGE)) < 1e-12
        # Flat stretches must genuinely vanish, not merely look small.
        assert abs(out[1]) < 0.3 and max(abs(v) for v in out) > 1.5

        self.play(FadeOut(self.out_bars), FadeOut(self.rows), run_time=0.5)
        new_kern = VGroup(
            VGroup(*[MathTex(rf"{w:+.0f}", font_size=24, color=EDGE_C)
                     for w in EDGE]).arrange(RIGHT, buff=0.3),
            VGroup(Text("sums to", font_size=18, color=DIM),
                   MathTex("0", font_size=24, color=EDGE_C)
                   ).arrange(RIGHT, buff=0.22),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        new_kern.move_to(self.kern)
        self.play(FadeTransform(self.kern, new_kern), run_time=0.8)
        self.kern = new_kern

        self.play(self.window.animate.set_stroke(EDGE_C), run_time=0.4)
        out_bars = self.bars(out, -1.9, EDGE_C)
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in out_bars],
                              lag_ratio=0.06, run_time=1.8))

        marks = VGroup(*[
            Circle(radius=0.2, color=EDGE_C, stroke_width=2.4).move_to(
                np.array([out_bars[i].get_center()[0], -1.9, 0.0]))
            for i, v in enumerate(out) if abs(v) > 1.5])
        self.play(LaggedStart(*[Create(m) for m in marks],
                              lag_ratio=0.2, run_time=1.2))
        note = Text("flat -> nothing, step -> spike", font_size=19,
                    color=EDGE_C).next_to(self.kern, DOWN, buff=0.5).align_to(
            self.kern, LEFT)
        self.play(FadeIn(note), run_time=0.8)
        self.edge_bars, self.marks, self.edge_note = out_bars, marks, note
        self.wait(1.4)

    @beat("Three numbers decide the whole job", seconds=24,
          narration="One more: minus a half, two, minus a half. This one "
                    "exaggerates whatever makes a value differ from its "
                    "neighbours, so the patches come back with harder edges "
                    "than they started with. Blur, edge detect, sharpen — one "
                    "operation, three behaviours, and the only thing that "
                    "changed was three numbers. That is what a convolutional "
                    "network learns: not the sliding, the weights.")
    def sharpen(self):
        out = convolve1d(SIGNAL, SHARP)
        assert abs(sum(SHARP) - 1.0) < 1e-12
        # It must genuinely exaggerate: wider spread than the input.
        spread_in = max(SIGNAL) - min(SIGNAL)
        spread_out = max(out) - min(out)
        assert spread_out > spread_in

        self.play(FadeOut(self.edge_bars), FadeOut(self.marks),
                  FadeOut(self.edge_note), run_time=0.5)
        new_kern = VGroup(
            VGroup(*[MathTex(rf"{w:+.1f}", font_size=24, color=OUT_C)
                     for w in SHARP]).arrange(RIGHT, buff=0.3),
            VGroup(Text("sums to", font_size=18, color=DIM),
                   MathTex("1", font_size=24, color=OUT_C)
                   ).arrange(RIGHT, buff=0.22),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        new_kern.move_to(self.kern)
        self.play(FadeTransform(self.kern, new_kern), run_time=0.8)
        self.play(self.window.animate.set_stroke(OUT_C), run_time=0.4)

        out_bars = self.bars(out, -1.9, OUT_C)
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in out_bars],
                              lag_ratio=0.06, run_time=1.8))

        rows = VGroup(
            VGroup(Text("spread in", font_size=18, color=DIM),
                   MathTex(rf"{spread_in:.2f}", font_size=22, color=SIG_C)
                   ).arrange(RIGHT, buff=0.22),
            VGroup(Text("spread out", font_size=18, color=DIM),
                   MathTex(rf"{spread_out:.2f}", font_size=22, color=OUT_C)
                   ).arrange(RIGHT, buff=0.22),
            Text("the weights are what is learned", font_size=17, color=DIM),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        rows.next_to(new_kern, DOWN, buff=0.5).align_to(new_kern, LEFT)
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.3, run_time=1.5))
        self.wait(2.2)
