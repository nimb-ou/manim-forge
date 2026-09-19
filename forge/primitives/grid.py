"""Grid primitives — chessboards, lattices, matrices, pixel grids.

The first entry in the verified-primitives library, and deliberately so: a
chessboard, a cellular-automaton lattice, a matrix and an image patch are the
same object wearing different clothes. One tested primitive covers all of them.

The governing principle of this package:

    **The animation computes its own claims.**

A scene that displays "P = 4/64" must derive that 4 and that 64 by counting
real cells, never by reciting a number a language model remembered. Everything
here returns exact values — ``Fraction``, not ``float`` — so the figure on
screen is correct by construction rather than by luck.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator


@dataclass(frozen=True, order=True)
class Cell:
    """A single grid cell, indexed from the bottom-left as (0, 0).

    Bottom-left origin matches both chess algebraic notation and Manim's
    y-up coordinate system, which saves a flip at render time.
    """
    row: int
    col: int

    @property
    def algebraic(self) -> str:
        """Chess-style name: (0,0) -> 'a1'. Only meaningful for boards <= 26 wide."""
        return f"{chr(ord('a') + self.col)}{self.row + 1}"

    def __repr__(self) -> str:
        return f"Cell({self.row},{self.col})"


class Board:
    """A rows x cols grid with exact, enumerable geometry.

    Every method counts real cells. Nothing is hardcoded, so the same code is
    correct for an 8x8 chessboard and a 3x5 lattice::

        >>> b = Board(8, 8)
        >>> len(b.corners()), b.n_cells
        (4, 64)
        >>> b.probability_of(b.corners())
        Fraction(1, 16)
    """

    def __init__(self, rows: int = 8, cols: int = 8) -> None:
        if rows < 1 or cols < 1:
            raise ValueError(f"Board must be at least 1x1, got {rows}x{cols}")
        self.rows = rows
        self.cols = cols

    # -- basic geometry ------------------------------------------------------

    @property
    def n_cells(self) -> int:
        return self.rows * self.cols

    def cells(self) -> Iterator[Cell]:
        for r in range(self.rows):
            for c in range(self.cols):
                yield Cell(r, c)

    def contains(self, cell: Cell) -> bool:
        return 0 <= cell.row < self.rows and 0 <= cell.col < self.cols

    def is_light(self, cell: Cell) -> bool:
        """Chessboard shading. a1 is dark, which is the real convention."""
        return (cell.row + cell.col) % 2 == 1

    # -- named subsets -------------------------------------------------------

    def corners(self) -> list[Cell]:
        """The corner cells. Four of them on any board bigger than 1x1 —
        deduplicated, so a 1xN board correctly reports 2, not 4."""
        candidates = {
            Cell(0, 0),
            Cell(0, self.cols - 1),
            Cell(self.rows - 1, 0),
            Cell(self.rows - 1, self.cols - 1),
        }
        return sorted(candidates)

    def edges(self) -> list[Cell]:
        """Cells on the boundary, corners included."""
        return sorted(
            c for c in self.cells()
            if c.row in (0, self.rows - 1) or c.col in (0, self.cols - 1)
        )

    def interior(self) -> list[Cell]:
        return sorted(set(self.cells()) - set(self.edges()))

    # -- piece movement ------------------------------------------------------

    def rook_moves(self, origin: Cell) -> list[Cell]:
        """Squares a rook reaches from ``origin`` on an empty board.

        Always ``(rows - 1) + (cols - 1)`` — 14 on a chessboard, from *every*
        square. That invariance is a genuinely interesting fact and worth
        animating; it falls out of counting rather than being asserted.
        """
        if not self.contains(origin):
            raise ValueError(f"{origin} is not on this {self.rows}x{self.cols} board")
        same_file = [Cell(r, origin.col) for r in range(self.rows) if r != origin.row]
        same_rank = [Cell(origin.row, c) for c in range(self.cols) if c != origin.col]
        return sorted(same_file + same_rank)

    def bishop_moves(self, origin: Cell) -> list[Cell]:
        """Diagonal reach. Unlike the rook's, this *does* depend on position —
        the contrast with ``rook_moves`` makes a good beat."""
        if not self.contains(origin):
            raise ValueError(f"{origin} is not on this board")
        out: list[Cell] = []
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            r, c = origin.row + dr, origin.col + dc
            while 0 <= r < self.rows and 0 <= c < self.cols:
                out.append(Cell(r, c))
                r += dr
                c += dc
        return sorted(out)

    def knight_moves(self, origin: Cell) -> list[Cell]:
        if not self.contains(origin):
            raise ValueError(f"{origin} is not on this board")
        deltas = ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2))
        return sorted(
            cell for dr, dc in deltas
            if self.contains(cell := Cell(origin.row + dr, origin.col + dc))
        )

    # -- exact probability ---------------------------------------------------

    def probability_of(self, subset) -> Fraction:
        """Exact probability of landing in ``subset`` under a uniform draw.

        Returns a ``Fraction``, never a float: the scene renders "1/16", and a
        binary-rounded 0.0625 has no business appearing in a maths explainer.
        Duplicates in ``subset`` are ignored, and membership is validated — a
        subset containing an off-board cell is a bug we want raised loudly,
        not silently folded into a wrong numerator.
        """
        unique = set(subset)
        for cell in unique:
            if not self.contains(cell):
                raise ValueError(f"{cell} is not on this {self.rows}x{self.cols} board")
        return Fraction(len(unique), self.n_cells)
