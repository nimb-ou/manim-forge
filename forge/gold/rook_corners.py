"""Gold scene 001 — rook mobility and corner probability.

Authored by hand from Nimit's own prompt:

    "build me a chess grid, add rooks on it and animate the rook's possible
     movements, then help me calculate the probability that if i place the
     rook at any square, it is a corner square"

This is a *target*, not a sample: the standard the fine-tuned model has to
reach. Two properties make it gold rather than merely working.

**Every number on screen is computed.** The 4, the 64 and the 1/16 come from
counting cells in ``forge.primitives.grid``. Nothing here recites a figure a
language model happened to remember.

**It answers more than it was asked.** The prompt requested rook movement and
corner probability as two separate things. Beat 4 finds the link worth showing:
a rook reaches exactly 14 squares from *every* square on the board, corners
included. That invariance came out of writing the primitive, and it is the
difference between a diagram and an explanation.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.grid import Board, Cell

# A restrained palette: neutral board, one colour for attention, one for the
# result. Loud boards fight the thing you are trying to point at.
LIGHT_SQ = "#3D4852"
DARK_SQ = "#2A313A"
ROOK_C = BLUE_C
REACH_C = YELLOW_C
CORNER_C = "#5CD0B3"

CELL = 0.70
BOARD_X = -3.1      # board centre; right half is the information panel
PANEL_X = 3.4


def rook_glyph(color=ROOK_C) -> VGroup:
    """A rook built from primitives rather than a font glyph — no typeface to
    be missing, and it scales cleanly."""
    base = Rectangle(width=0.40, height=0.07, fill_opacity=1, color=color, stroke_width=0)
    stem = Rectangle(width=0.24, height=0.24, fill_opacity=1, color=color, stroke_width=0)
    head = Rectangle(width=0.36, height=0.10, fill_opacity=1, color=color, stroke_width=0)
    stem.next_to(base, UP, buff=0)
    head.next_to(stem, UP, buff=0)
    return VGroup(base, stem, head)


class RookCorners(ForgeScene):

    board = Board(8, 8)

    # -- geometry helpers ----------------------------------------------------

    def xy(self, cell: Cell) -> np.ndarray:
        """Cell -> world coordinates, with the board centred on the origin."""
        x = BOARD_X + (cell.col - (self.board.cols - 1) / 2) * CELL
        y = (cell.row - (self.board.rows - 1) / 2) * CELL - 0.3
        return np.array([x, y, 0.0])

    def tile(self, cell: Cell, color, opacity=0.85) -> Square:
        return Square(side_length=CELL, stroke_width=0,
                      fill_color=color, fill_opacity=opacity).move_to(self.xy(cell))

    # -- beats ---------------------------------------------------------------

    @beat("Draw an 8x8 chessboard", seconds=3)
    def draw_board(self):
        self.squares = VGroup(*[
            self.tile(c, LIGHT_SQ if self.board.is_light(c) else DARK_SQ, 1.0)
            for c in self.board.cells()
        ])
        self.title = Text("A rook on an 8 x 8 board", font_size=30).to_edge(UP, buff=0.35)
        self.play(LaggedStart(*[FadeIn(s, scale=0.6) for s in self.squares],
                              lag_ratio=0.012, run_time=2.0))
        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)

    @beat("Place a rook on d4", seconds=2)
    def place_rook(self):
        self.origin = Cell(3, 3)                      # d4
        self.rook = rook_glyph().move_to(self.xy(self.origin))
        self.play(FadeIn(self.rook, shift=UP * 0.4), run_time=0.8)
        self.play(Flash(self.rook, color=ROOK_C, line_length=0.16, num_lines=10,
                        flash_radius=0.42), run_time=0.6)

    @beat("Highlight every square the rook can reach", seconds=5)
    def show_reach(self):
        reach = self.board.rook_moves(self.origin)     # computed, not hardcoded
        self.reach_tiles = VGroup(*[self.tile(c, REACH_C, 0.42) for c in reach])

        self.count = VGroup(
            Text(str(len(reach)), font_size=72, color=REACH_C),
            Text("reachable squares", font_size=24, color=GREY_B),
        ).arrange(DOWN, buff=0.18).move_to([PANEL_X, 0.6, 0])

        self.play(LaggedStart(*[FadeIn(t) for t in self.reach_tiles],
                              lag_ratio=0.05, run_time=1.8))
        self.play(FadeIn(self.count[0], scale=0.7), Write(self.count[1]), run_time=0.8)
        self.wait(0.4)

    @beat("Move the rook to the corner - the count is unchanged", seconds=5)
    def corner_is_no_different(self):
        corner = Cell(0, 0)                            # a1
        reach = self.board.rook_moves(corner)
        new_tiles = VGroup(*[self.tile(c, REACH_C, 0.42) for c in reach])

        self.play(
            self.rook.animate.move_to(self.xy(corner)),
            Transform(self.reach_tiles, new_tiles),
            run_time=1.6,
        )
        # The count is unchanged, so the numeral must visibly hold still while
        # everything around it moves — that stillness *is* the point of the beat.
        self.play(Indicate(self.count[0], color=REACH_C, scale_factor=1.25), run_time=0.8)

        note = VGroup(
            Text("still 14.", font_size=28, color=REACH_C),
            Text("a rook's reach is the", font_size=22, color=GREY_B),
            Text("same from every square", font_size=22, color=GREY_B),
        ).arrange(DOWN, buff=0.12).move_to([PANEL_X, -1.5, 0])
        self.play(FadeIn(note, shift=UP * 0.15), run_time=0.8)
        self.wait(0.9)
        self.play(FadeOut(note), FadeOut(self.reach_tiles), FadeOut(self.count),
                  run_time=0.6)

    @beat("Highlight the four corner squares", seconds=4)
    def show_corners(self):
        corners = self.board.corners()                 # computed -> 4
        self.corner_tiles = VGroup(*[self.tile(c, CORNER_C, 0.55) for c in corners])

        new_title = Text("How likely is a corner?", font_size=30).to_edge(UP, buff=0.35)
        self.play(Transform(self.title, new_title), run_time=0.8)
        self.play(FadeOut(self.rook, shift=DOWN * 0.3), run_time=0.5)
        self.play(LaggedStart(*[GrowFromCenter(t) for t in self.corner_tiles],
                              lag_ratio=0.15, run_time=1.4))
        self.wait(0.3)

    @beat("Derive the probability as an exact fraction", seconds=6)
    def derive(self):
        corners = self.board.corners()
        p = self.board.probability_of(corners)         # Fraction(1, 16) - exact

        expr = MathTex(
            r"P(\text{corner})", "=",
            rf"\frac{{{len(corners)}}}{{{self.board.n_cells}}}",
            font_size=42,
        ).move_to([PANEL_X, 0.9, 0])
        expr[2].set_color(CORNER_C)

        reduced = MathTex("=", rf"\frac{{{p.numerator}}}{{{p.denominator}}}", font_size=52)
        reduced[1].set_color(CORNER_C)
        reduced.next_to(expr, DOWN, buff=0.45)

        self.play(Write(expr[0]), run_time=0.6)
        self.play(Write(expr[1]), Write(expr[2]), run_time=1.0)
        self.wait(0.5)
        self.play(Write(reduced), run_time=0.9)

        pct = Text(f"{float(p) * 100:.2f}%", font_size=26, color=GREY_B)
        pct.next_to(reduced, DOWN, buff=0.4)
        self.play(FadeIn(pct, shift=UP * 0.12), run_time=0.6)
        self.expr, self.reduced, self.pct = expr, reduced, pct
        self.wait(1.0)
