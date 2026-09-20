"""Gold scene 030 — belief as area.

Probability, Tier 2. The medical test is the case where everyone's intuition
fails the same way, and it fails because the question gets swapped: "how often
does the test catch the disease" is not "how likely am I to have it".

The scene works in counts, not rates. Ten thousand people, drawn as a block of
squares, split by who is ill and then by who tests positive -- because the
whole difficulty is that a small slice of a large group can outnumber most of
a small one, and that is a fact about areas, visible on sight.

``verify_against_bayes`` checks the counted answer against the formula, so the
picture and the algebra cannot drift apart.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.probability import TestOutcome

SICK_C = "#FC6255"
WELL_C = GREY_D
TP_C = "#F0AC5F"
FP_C = "#5CD0B3"
DIM = GREY_B

T = TestOutcome(population=10000, prevalence=0.01,
                sensitivity=0.99, specificity=0.95)

COLS, ROWS = 100, 100
BLOCK_W, BLOCK_H = 4.6, 4.0
BLOCK_AT = LEFT * 3.3 + DOWN * 0.35


class Bayes(ForgeScene):

    def cell_rect(self, i0, n, colour, opacity=0.8):
        """A run of ``n`` people, as a band of the block."""
        cw = BLOCK_W / COLS
        x0 = BLOCK_AT[0] - BLOCK_W / 2 + (i0 / T.population) * BLOCK_W
        w = (n / T.population) * BLOCK_W
        return Rectangle(width=max(w, 0.012), height=BLOCK_H, color=colour,
                         fill_color=colour, fill_opacity=opacity,
                         stroke_width=0).move_to(
            np.array([x0 + w / 2, BLOCK_AT[1], 0.0]))

    def panel(self, *rows):
        g = VGroup(*rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        return g.move_to(RIGHT * 3.2).align_to(UP * 2.3, UP)

    @beat("A test that is right almost every time", seconds=17,
          narration="A disease that one person in a hundred has. A test that "
                    "catches ninety-nine percent of the people who have it, "
                    "and correctly clears ninety-five percent of those who do "
                    "not. You take it. It comes back positive. How worried "
                    "should you be?")
    def setup(self):
        self.title = Text("What a positive test actually tells you",
                          font_size=28).to_edge(UP, buff=0.4)
        self.block = Rectangle(width=BLOCK_W, height=BLOCK_H, color=GREY_B,
                               stroke_width=2).move_to(BLOCK_AT)
        cap = Text(f"{T.population:,} people", font_size=19, color=DIM).next_to(
            self.block, DOWN, buff=0.25)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(Create(self.block), FadeIn(cap), run_time=1.2)

        facts = self.panel(
            VGroup(Text("have it", font_size=19, color=DIM),
                   MathTex(r"1\%", font_size=26, color=SICK_C)
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
            VGroup(Text("test catches", font_size=19, color=DIM),
                   MathTex(r"99\%", font_size=26, color=TP_C)
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
            VGroup(Text("correctly clears", font_size=19, color=DIM),
                   MathTex(r"95\%", font_size=26, color=FP_C)
                   ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN),
            Text("you test positive.", font_size=21, color=WHITE),
        )
        self.play(LaggedStart(*[FadeIn(f, shift=LEFT * 0.2) for f in facts],
                              lag_ratio=0.3, run_time=2.2))
        self.facts, self.cap = facts, cap
        self.wait(1.4)

    @beat("Split the crowd by who is actually ill", seconds=17,
          narration="Take ten thousand people. One in a hundred means a "
                    "hundred of them are ill — that thin sliver on the left. "
                    "The other nine thousand nine hundred are not. Notice how "
                    "small the sliver is. Everything that follows comes from "
                    "that.")
    def split(self):
        assert T.sick == 100 and T.well == 9900

        self.play(FadeOut(self.facts), run_time=0.3)
        sick = self.cell_rect(0, T.sick, SICK_C, 0.9)
        well = self.cell_rect(T.sick, T.well, WELL_C, 0.55)
        self.play(FadeIn(well), run_time=0.9)
        self.play(FadeIn(sick), run_time=0.9)

        labs = self.panel(
            VGroup(MathTex(rf"{T.sick}", font_size=30, color=SICK_C),
                   Text("ill", font_size=20, color=DIM)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(MathTex(rf"{T.well:,}", font_size=30, color=WELL_C),
                   Text("not ill", font_size=20, color=DIM)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
        )
        self.play(LaggedStart(*[FadeIn(l, shift=LEFT * 0.2) for l in labs],
                              lag_ratio=0.35, run_time=1.4))
        self.sick_band, self.well_band, self.labs = sick, well, labs
        self.wait(1.4)

    @beat("Now count everyone the test calls positive", seconds=25,
          narration="Run the test on all of them. Ninety-nine of the hundred "
                    "ill people come back positive — the test is very good. "
                    "But five percent of nine thousand nine hundred healthy "
                    "people also come back positive, and five percent of a "
                    "very large number is four hundred and ninety-five. Far "
                    "more false alarms than real cases, from a test that is "
                    "almost never wrong.")
    def positives(self):
        assert T.true_positive == 99 and T.false_positive == 495

        self.play(FadeOut(self.labs), run_time=0.3)
        tp = self.cell_rect(0, T.true_positive, TP_C, 0.95)
        fp = self.cell_rect(T.sick, T.false_positive, FP_C, 0.8)
        self.play(self.sick_band.animate.set_opacity(0.25),
                  self.well_band.animate.set_opacity(0.2), run_time=0.7)
        self.play(FadeIn(tp), run_time=0.9)
        self.play(FadeIn(fp), run_time=1.1)

        counts = self.panel(
            VGroup(MathTex(rf"{T.true_positive}", font_size=30, color=TP_C),
                   Text("ill, positive", font_size=19, color=DIM)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(MathTex(rf"{T.false_positive}", font_size=30, color=FP_C),
                   Text("well, positive", font_size=19, color=DIM)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
            VGroup(MathTex(rf"{T.positive}", font_size=26, color=WHITE),
                   Text("positive in total", font_size=19, color=DIM)
                   ).arrange(RIGHT, buff=0.25, aligned_edge=DOWN),
        )
        self.play(LaggedStart(*[FadeIn(c, shift=LEFT * 0.2) for c in counts],
                              lag_ratio=0.35, run_time=1.8))
        self.tp, self.fp, self.counts = tp, fp, counts
        self.wait(1.6)

    @beat("A positive result means one chance in six", seconds=26,
          narration="So of the five hundred and ninety-four people holding a "
                    "positive result, only ninety-nine are actually ill. That "
                    "is sixteen point seven percent — about one in six. The "
                    "test did not lie and the arithmetic is not a trick. A "
                    "small share of a very large group simply outnumbers most "
                    "of a very small one, which is what the picture shows and "
                    "what intuition refuses to.")
    def answer(self):
        assert T.verify_against_bayes()
        assert round(T.posterior * 100, 1) == 16.7
        assert T.positive == 594

        self.play(FadeOut(self.counts), FadeOut(self.sick_band),
                  FadeOut(self.well_band), run_time=0.5)
        self.play(self.tp.animate.set_opacity(1.0),
                  self.fp.animate.set_opacity(0.9), run_time=0.7)

        frac = VGroup(
            MathTex(rf"\frac{{{T.true_positive}}}{{{T.positive}}}",
                    font_size=44, color=TP_C),
            MathTex(rf"= {T.posterior:.1%}".replace("%", r"\%"),
                    font_size=36, color=WHITE),
        ).arrange(RIGHT, buff=0.35)
        frac.move_to(RIGHT * 3.3 + UP * 1.4)
        self.play(Write(frac[0]), run_time=1.2)
        self.play(FadeIn(frac[1], shift=LEFT * 0.15), run_time=0.8)

        rule = VGroup(
            Text("about one in six", font_size=24, color=TP_C),
            Text("counted, then checked", font_size=17, color=DIM),
            Text("against Bayes' rule", font_size=17, color=DIM),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        rule.next_to(frac, DOWN, buff=0.7).align_to(frac, LEFT)
        self.play(FadeIn(rule, shift=UP * 0.12), run_time=1.0)
        self.wait(2.4)
