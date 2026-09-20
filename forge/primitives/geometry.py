"""Geometry primitives — rearrangement proofs and limiting shapes.

Both scenes that use this module make the same kind of claim: that two
pictures have the same area, or that a sequence of pictures approaches one.
Those are claims a drawing can fake perfectly, so they are computed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RightTriangle:
    """A right triangle, and the squares built on its sides."""
    a: float
    b: float

    @property
    def c(self) -> float:
        return math.hypot(self.a, self.b)

    @property
    def area(self) -> float:
        return self.a * self.b / 2.0

    def outer_square_side(self) -> float:
        """The (a+b) square that four copies of this triangle fit inside."""
        return self.a + self.b

    def leftover_two_squares(self) -> float:
        """Area left when the four triangles are packed into two corners."""
        return self.outer_square_side() ** 2 - 4 * self.area

    def leftover_one_square(self) -> float:
        """Area left when the same four are packed as a pinwheel."""
        return self.outer_square_side() ** 2 - 4 * self.area

    def verify_rearrangement(self, tol: float = 1e-9) -> bool:
        """Both packings leave the same area, and it is a^2+b^2 and c^2.

        The two arrangements use identical pieces inside identical squares, so
        what is left must match. That the leftover equals both a^2+b^2 and c^2
        is the theorem; checking it here means the scene cannot state it while
        drawing a triangle where it does not hold.
        """
        left = self.leftover_two_squares()
        return (abs(left - self.leftover_one_square()) < tol
                and abs(left - (self.a ** 2 + self.b ** 2)) < tol
                and abs(left - self.c ** 2) < tol)


@dataclass
class SectorRearrangement:
    """A circle cut into ``n`` sectors and interleaved into a near-rectangle."""
    n: int
    r: float = 1.0

    @property
    def half_angle(self) -> float:
        return math.pi / self.n

    @property
    def width(self) -> float:
        """Base of the parallelogram: half the arcs, laid end to end.

        The *arc* length, not the chord. Using the chord would make the shape
        converge to the wrong width and quietly understate the area.
        """
        return math.pi * self.r

    @property
    def height(self) -> float:
        """Height of the parallelogram: the apothem, not the radius.

        This is where the approximation lives. Each sector's straight height
        is r*cos(pi/n), which rises to r only in the limit -- so a scene that
        labels the height r for small n is drawing a shape it has not got.
        """
        return self.r * math.cos(self.half_angle)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def exact(self) -> float:
        return math.pi * self.r ** 2

    @property
    def error(self) -> float:
        return self.exact - self.area


def sector_convergence(ns: list[int], r: float = 1.0
                       ) -> list[tuple[int, float, float]]:
    """(n, area of the rearranged shape, error against pi r^2)."""
    return [(n, SectorRearrangement(n, r).area, SectorRearrangement(n, r).error)
            for n in ns]


def verify_sectors_converge(r: float = 1.0, n: int = 4096,
                            tol: float = 1e-3) -> bool:
    """The rearranged area must approach pi r^2 as the sectors get thin."""
    return abs(SectorRearrangement(n, r).error) < tol
