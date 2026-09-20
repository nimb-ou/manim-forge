"""Gold scene 001 — rook mobility and corner probability.

Authored by hand from Nimit's own prompt:

    "build a chess grid, add rooks on it and animate the rook's possible
     movements, then help me calculate the probability that if i place the
     rook at any square, it is a corner square"

Rewritten for logical flow. The first version showed rook movement, then
showed corner probability, and never joined them — two facts standing next to
each other rather than an argument. This version asks one question and follows
it: *does it matter where the rook stands?* Reach says no, and says it
emphatically, from three different squares. Then the corner question says yes,
and quantifies it. The pair is the point.

Every number is computed from ``forge.primitives.grid``. The 14 comes from
counting reachable squares, the 4 and 64 from counting cells, and the 1/16 is
an exact Fraction derived from those two.
"""

from fractions import Fraction

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.grid import Board, Cell

LIGHT_SQ = "#3D4852"
DARK_SQ = "#2A313A"
ROOK_C = BLUE_C
REACH_C = YELLOW_C
CORNER_C = "#5CD0B3"
DIM = GREY_B

CELL = 0.66
BOARD_X = -3.2
PANEL_X = 3.5

# Slower and eased throughout: the first cut moved between states faster than
# a viewer can follow what changed, which is most of what made it feel thin.
EASE = rate_functions.ease_in_out_sine


def rook_glyph(color=ROOK_C) -> VGroup:
    """A rook from primitives rather than a font glyph — no typeface to be
    missing, and it scales cleanly."""
    base = Rectangle(width=0.38, height=0.07, fill_opacity=1, color=color, stroke_width=0)
    stem = Rectangle(width=0.22, height=0.22, fill_opacity=1, color=color, stroke_width=0)
    head = Rectangle(width=0.34, height=0.09, fill_opacity=1, color=color, stroke_width=0)
    stem.next_to(base, UP, buff=0)
    head.next_to(stem, UP, buff=0)
    return VGroup(base, stem, head)


class RookCorners(ForgeScene):

    board = Board(8, 8)

    # -- geometry ------------------------------------------------------------

    def xy(self, cell: Cell) -> np.ndarray:
        x = BOARD_X + (cell.col - (self.board.cols - 1) / 2) * CELL
        y = (cell.row - (self.board.rows - 1) / 2) * CELL - 0.25
        return np.array([x, y, 0.0])

    def tile(self, cell: Cell, color, opacity=0.85) -> Square:
        return Square(side_length=CELL, stroke_width=0,
                      fill_color=color, fill_opacity=opacity).move_to(self.xy(cell))

    def reach_tiles(self, origin: Cell) -> VGroup:
        return VGroup(*[self.tile(c, REACH_C, 0.4)
                        for c in self.board.rook_moves(origin)])

    def counter(self, n: int, caption: str, color=REACH_C, y=0.7) -> VGroup:
        return VGroup(
            Text(str(n), font_size=76, color=color),
            Text(caption, font_size=22, color=DIM),
        ).arrange(DOWN, buff=0.16).move_to([PANEL_X, y, 0])

    # -- beats ---------------------------------------------------------------

    @beat("Lay out the board and drop a rook on it", seconds=11,
          narration="Here is a chessboard, and a rook sitting somewhere on it. "
                    "I want to ask a simple question about this piece: does it "
                    "matter where it stands?")
    def setup(self):
        self.squares = VGroup(*[
            self.tile(c, LIGHT_SQ if self.board.is_light(c) else DARK_SQ, 1.0)
            for c in self.board.cells()])
        self.title = Text("Does it matter where the rook stands?",
                          font_size=28).to_edge(UP, buff=0.4)

        self.origin = Cell(4, 2)                     # c5
        self.rook = rook_glyph().move_to(self.xy(self.origin))

        self.play(LaggedStart(*[FadeIn(s, scale=0.55) for s in self.squares],
                              lag_ratio=0.011, run_time=2.2))
        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.9)
        self.play(FadeIn(self.rook, shift=UP * 0.45, rate_func=EASE), run_time=1.0)
        self.wait(0.5)

    @beat("Count the squares it can reach", seconds=9,
          narration="A rook moves along its rank and its file. Light those up "
                    "and count them: fourteen squares it can reach from here.")
    def first_count(self):
        reach = self.board.rook_moves(self.origin)   # computed
        self.tiles = self.reach_tiles(self.origin)
        self.count = self.counter(len(reach), "squares it can reach")

        self.play(LaggedStart(*[FadeIn(t) for t in self.tiles],
                              lag_ratio=0.045, run_time=2.0))
        self.play(FadeIn(self.count[0], scale=0.7), Write(self.count[1]),
                  run_time=1.0)
        self.wait(0.8)

    @beat("Move it twice more - the count refuses to change", seconds=14,
          narration="Now move it. To the edge — still fourteen. Into the "
                    "corner, the most cramped square on the board — still "
                    "fourteen. Whatever you do to a rook's position, its reach "
                    "does not care.")
    def unchanged(self):
        for target in (Cell(7, 5), Cell(0, 0)):      # edge, then corner
            reach = self.board.rook_moves(target)
            assert len(reach) == len(self.board.rook_moves(self.origin))
            fresh = self.reach_tiles(target)
            # Fade out and in rather than Transform. Transforming one VGroup of
            # squares into another makes Manim pair them up and morph each into
            # its neighbour's position, which smears across the whole board
            # mid-animation. The rook glides; the highlight simply changes.
            self.play(FadeOut(self.tiles, run_time=0.5),
                      self.rook.animate(rate_func=EASE, run_time=1.2).move_to(self.xy(target)))
            self.tiles = fresh
            self.play(LaggedStart(*[FadeIn(t) for t in self.tiles],
                                  lag_ratio=0.03, run_time=1.0))
            # The numeral holding still while everything moves *is* the point,
            # so it is pulsed rather than replaced.
            self.play(Indicate(self.count[0], color=REACH_C, scale_factor=1.2),
                      run_time=0.8)
            self.wait(0.5)

        self.verdict = Text("the reach never changes", font_size=24, color=REACH_C)
        self.verdict.move_to([PANEL_X, -0.9, 0])
        self.play(FadeIn(self.verdict, shift=UP * 0.15), run_time=0.8)
        self.wait(1.0)

    @beat("So ask the opposite question", seconds=16,
          narration="So position does not matter — at least not for reach. But "
                    "some squares really are different from others. The four "
                    "corners are the most cramped places on the board. How "
                    "likely is the rook to be standing on one?")
    def pivot(self):
        new_title = Text("So how often does it land in a corner?",
                         font_size=28).to_edge(UP, buff=0.4)
        self.play(
            FadeOut(self.tiles), FadeOut(self.count), FadeOut(self.verdict),
            run_time=0.8,
        )
        self.play(Transform(self.title, new_title), run_time=1.0)

        corners = self.board.corners()               # computed -> 4
        self.corner_tiles = VGroup(*[self.tile(c, CORNER_C, 0.55) for c in corners])
        self.play(LaggedStart(*[GrowFromCenter(t) for t in self.corner_tiles],
                              lag_ratio=0.18, run_time=1.5))
        self.wait(0.4)

    @beat("Count both sets and divide", seconds=15,
          narration="Four corners. Sixty-four squares in total. So the chance "
                    "is four in sixty-four, which is one in sixteen — a little "
                    "over six percent. One question about this board ignores "
                    "position entirely. The other is nothing but position.")
    def derive(self):
        corners = self.board.corners()
        p = self.board.probability_of(corners)       # exact Fraction
        assert p == Fraction(1, 16)

        four = self.counter(len(corners), "corner squares", CORNER_C, y=1.6)
        self.play(FadeIn(four[0], scale=0.7), Write(four[1]), run_time=0.9)
        self.wait(0.4)

        sixtyfour = self.counter(self.board.n_cells, "squares in total", DIM, y=1.6)
        self.play(ReplacementTransform(four, sixtyfour), run_time=1.0)
        self.play(Indicate(self.squares, color=DIM, scale_factor=1.015), run_time=1.0)
        self.wait(0.3)

        expr = MathTex(
            rf"\frac{{{len(corners)}}}{{{self.board.n_cells}}}", "=",
            rf"\frac{{{p.numerator}}}{{{p.denominator}}}", font_size=52,
        ).move_to([PANEL_X, 0.1, 0])
        expr[0].set_color(CORNER_C)
        expr[2].set_color(CORNER_C)
        pct = Text(f"{float(p)*100:.2f}%", font_size=26, color=DIM)
        pct.next_to(expr, DOWN, buff=0.45)

        self.play(FadeOut(sixtyfour), run_time=0.4)
        self.play(Write(expr), run_time=1.4)
        self.play(FadeIn(pct, shift=UP * 0.12), run_time=0.7)
        self.wait(1.6)
