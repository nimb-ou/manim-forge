"""Calculus primitives — Riemann sums, derivatives, and convergence.

Fifth in the verified-primitives library, same rule: the animation computes its
own claims. A scene showing Riemann rectangles converging must compute the sums
it draws and the limit it converges to, so the numbers on screen are facts
about the function rather than figures chosen to look convincing.

The convergence is the point of such a scene, and it is a real numerical fact —
so the scene can honestly show the error shrinking, and would show it *not*
shrinking if the maths were wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

Rule = Literal["left", "right", "midpoint"]


@dataclass
class Rectangle:
    """One Riemann rectangle: where it sits and how tall it is."""
    x_left: float
    x_right: float
    height: float

    @property
    def width(self) -> float:
        return self.x_right - self.x_left

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class RiemannSum:
    """A Riemann approximation of ∫f over [a, b] with n subintervals.

    ``rule`` selects the sample point. Midpoint is far more accurate than left
    or right for smooth functions — its error falls as n² rather than n — and
    contrasting them makes a better scene than showing one in isolation.
    """
    f: Callable[[float], float]
    a: float
    b: float
    n: int
    rule: Rule = "left"

    @property
    def dx(self) -> float:
        return (self.b - self.a) / self.n

    def rectangles(self) -> list[Rectangle]:
        out = []
        for i in range(self.n):
            xl = self.a + i * self.dx
            xr = xl + self.dx
            if self.rule == "left":
                sample = xl
            elif self.rule == "right":
                sample = xr
            else:
                sample = (xl + xr) / 2
            out.append(Rectangle(xl, xr, self.f(sample)))
        return out

    @property
    def total(self) -> float:
        return sum(r.area for r in self.rectangles())


def exact_integral(f: Callable[[float], float], a: float, b: float,
                   steps: int = 200_000) -> float:
    """A reference value by very fine midpoint summation.

    Numerical rather than symbolic so any callable works, and fine enough that
    the error is far below anything a scene displays. This is what the
    approximations are honestly compared against.
    """
    h = (b - a) / steps
    return sum(f(a + (i + 0.5) * h) for i in range(steps)) * h


def convergence(f: Callable[[float], float], a: float, b: float,
                counts: list[int], rule: Rule = "left") -> list[tuple[int, float, float]]:
    """(n, approximation, absolute error) for each n — the story of the scene."""
    exact = exact_integral(f, a, b)
    return [(n, s.total, abs(s.total - exact))
            for n in counts
            for s in (RiemannSum(f, a, b, n, rule),)]
