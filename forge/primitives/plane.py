"""Coordinate-plane primitives — addresses, distances, and loci.

Same rule as the rest of the library: the animation computes its own claims.
A scene that labels a dot ``(3, 1)`` must read those numbers off the position
it actually drew, and a scene claiming twelve lattice points sit exactly five
units from the origin must find them rather than be told.

The distance case is the one that bites. Pythagoras is easy to state and easy
to draw slightly wrong -- a hypotenuse sketched to look right, with a length
typed in beside it -- and nothing on screen would give it away.
"""

from __future__ import annotations

import math

from forge.primitives.vectors import Vec2


def distance(a: Vec2, b: Vec2) -> float:
    """Straight-line distance between two points."""
    return (a - b).norm          # Vec2.norm is a property, not a method


def legs(a: Vec2, b: Vec2) -> tuple[float, float]:
    """The horizontal and vertical legs of the right triangle on ``a``-``b``.

    Returned as unsigned lengths, which is what a scene labels them with; the
    sign lives in the geometry, not in the measurement.
    """
    return abs(b.x - a.x), abs(b.y - a.y)


def verify_pythagoras(a: Vec2, b: Vec2, tol: float = 1e-9) -> bool:
    """The legs and the distance must satisfy the theorem the scene invokes."""
    dx, dy = legs(a, b)
    return abs(dx * dx + dy * dy - distance(a, b) ** 2) < tol


def integer_points_on_circle(r: int) -> list[Vec2]:
    """Every lattice point at *exactly* distance ``r`` from the origin.

    Found by search over the bounding box and an exact integer test, never by
    floating-point comparison: ``x*x + y*y == r*r`` in integers is decidable,
    while ``math.hypot(x, y) == r`` is a coin toss at the boundary.

    r = 5 gives twelve, which is why five is the radius every textbook picks.
    """
    out: list[Vec2] = []
    for x in range(-r, r + 1):
        for y in range(-r, r + 1):
            if x * x + y * y == r * r:
                out.append(Vec2(float(x), float(y)))
    return out


def verify_on_circle(pts: list[Vec2], r: float, tol: float = 1e-9) -> bool:
    """Every listed point really is on the circle the scene draws."""
    return all(abs(p.norm - r) < tol for p in pts)


def circle_points(n: int, r: float = 1.0) -> list[Vec2]:
    """``n`` points evenly spaced around a circle, starting at angle zero."""
    return [Vec2(r * math.cos(2 * math.pi * i / n),
                 r * math.sin(2 * math.pi * i / n)) for i in range(n)]
