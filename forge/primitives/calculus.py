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


# ------------------------------------------------- derivatives and Taylor


def secant_slope(f: Callable[[float], float], x: float, h: float) -> float:
    """Slope of the line through (x, f(x)) and (x+h, f(x+h))."""
    return (f(x + h) - f(x)) / h


def secant_sequence(f: Callable[[float], float], x: float,
                    hs: list[float]) -> list[tuple[float, float]]:
    """(h, slope) as h shrinks -- the sequence the limit is taken along.

    Returned as data so a scene animates the *actual* convergence rather than
    a smooth interpolation towards a known answer. The whole idea of the
    derivative is that this sequence settles; staging it would skip the point.
    """
    return [(h, secant_slope(f, x, h)) for h in hs]


def numeric_derivative(f: Callable[[float], float], x: float,
                       h: float = 1e-6) -> float:
    """Central difference: error falls as h^2 rather than h.

    Used to *check* an analytic derivative, never to display one -- finite
    differences are accurate enough to catch a wrong formula and not accurate
    enough to be quoted.
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def verify_derivative(f: Callable[[float], float],
                      fprime: Callable[[float], float],
                      xs: list[float], tol: float = 1e-5) -> bool:
    """An analytic derivative must agree with the numeric one everywhere tested."""
    return all(abs(fprime(x) - numeric_derivative(f, x)) < tol for x in xs)


def taylor_poly(derivs: list[Callable[[float], float]], a: float
                ) -> Callable[[float], float]:
    """The Taylor polynomial about ``a`` built from the supplied derivatives.

    ``derivs[0]`` is f itself, ``derivs[k]`` its k-th derivative. Coefficients
    are f^(k)(a)/k!, computed here rather than written down, so adding a term
    to a scene cannot silently use the wrong factorial.
    """
    import math as _m
    coeffs = [d(a) / _m.factorial(k) for k, d in enumerate(derivs)]

    def p(x: float) -> float:
        return sum(c * (x - a) ** k for k, c in enumerate(coeffs))
    return p


def taylor_error(f: Callable[[float], float],
                 derivs: list[Callable[[float], float]],
                 a: float, x: float) -> float:
    """How far the polynomial misses at ``x``. Signed, because the sign matters."""
    return taylor_poly(derivs, a)(x) - f(x)


def verify_taylor_improves(f: Callable[[float], float],
                           derivs: list[Callable[[float], float]],
                           a: float, x: float) -> bool:
    """Each extra term must bring the approximation closer at ``x``.

    Not a general truth about Taylor series -- it fails outside the radius of
    convergence, and that is exactly why a scene claiming "more terms, better
    fit" has to check it on the interval it actually draws.
    """
    errs = [abs(taylor_error(f, derivs[:k + 1], a, x))
            for k in range(len(derivs))]
    return all(b <= a_ + 1e-12 for a_, b in zip(errs, errs[1:]))


@dataclass
class ChainLink:
    """One stage of a composition, with its own rate."""
    name: str
    value: float
    rate: float


def chain_trace(inner: Callable[[float], float],
                inner_prime: Callable[[float], float],
                outer: Callable[[float], float],
                outer_prime: Callable[[float], float],
                x: float) -> tuple[ChainLink, ChainLink, float]:
    """Each stage of f(g(x)) with its local rate, and the product.

    Returned as stages rather than a single number because the chain rule is a
    statement about stages: how fast the inside moves, times how fast the
    outside moves *at the value the inside currently has*. The second half is
    the part that gets dropped, and it is the only part that is subtle.
    """
    u = inner(x)
    g = ChainLink("inner", u, inner_prime(x))
    f = ChainLink("outer", outer(u), outer_prime(u))
    return g, f, g.rate * f.rate


def verify_chain_rule(inner, inner_prime, outer, outer_prime,
                      xs: list[float], tol: float = 1e-5) -> bool:
    """The product of rates must equal the numeric derivative of the composite.

    This is the check that catches the classic error: evaluating the outer
    derivative at x instead of at g(x). That mistake gives a plausible number,
    and on many functions it is even close.
    """
    def composite(t):
        return outer(inner(t))
    for x in xs:
        _, _, product = chain_trace(inner, inner_prime, outer, outer_prime, x)
        if abs(product - numeric_derivative(composite, x)) > tol:
            return False
    return True
