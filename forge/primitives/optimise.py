"""Optimisation primitives — gradient descent and friends.

Second entry in the verified-primitives library, and it follows the same rule
as ``grid``: **the animation computes its own claims.**

A scene about gradient descent must not draw a dot at hand-picked coordinates
and label it "step 3". It must actually run the update rule and animate where
the optimiser really went. Then the path on screen is a fact about the maths
rather than an illustration of it — and when the learning rate is too large,
the animation diverges because the *algorithm* diverges.

Derivatives are central differences rather than symbolic: it works for any
callable the caller supplies, including ones with no closed form, and the error
is far below one pixel at these scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Step:
    x: float
    y: float          # f(x)
    grad: float
    next_x: float


@dataclass
class Descent:
    """A gradient-descent run, recorded step by step.

    ``steps`` is the ground truth the animation replays. Nothing about the
    trajectory is authored — change the learning rate and the picture changes
    because the arithmetic does.
    """
    f: Callable[[float], float]
    x0: float
    lr: float
    n_steps: int = 12
    steps: list[Step] = field(default_factory=list)

    def __post_init__(self) -> None:
        x = float(self.x0)
        for _ in range(self.n_steps):
            g = self.gradient(x)
            nxt = x - self.lr * g
            self.steps.append(Step(x=x, y=self.f(x), grad=g, next_x=nxt))
            if not _finite(nxt) or abs(nxt) > 1e6:
                break            # diverged: stop, and let the scene show it
            x = nxt

    def gradient(self, x: float, h: float = 1e-5) -> float:
        return (self.f(x + h) - self.f(x - h)) / (2 * h)

    @property
    def converged(self) -> bool:
        """Did it settle? Judged by the last step barely moving.

        Tolerance is 1e-2 in x, which is well under a pixel once the axis is
        drawn — a tighter bound reports "not converged" for a trajectory that
        is visually sitting still, which would make the scene lie.
        """
        if len(self.steps) < 2:
            return False
        return abs(self.steps[-1].next_x - self.steps[-1].x) < 1e-2

    @property
    def diverged(self) -> bool:
        """Is it running away?

        Overflow alone is the wrong test. With a gentle curvature an unstable
        learning rate grows steadily without reaching 1e6 in a dozen steps, so
        the run looked stable while every step was getting larger. Growth in
        step size is the honest signal, and it is what the "learning rate too
        large" beat depends on being right.
        """
        if not self.steps:
            return False
        if not _finite(self.steps[-1].next_x):
            return True
        if len(self.steps) < 3:
            return False
        first = abs(self.steps[1].x - self.steps[0].x)
        last = abs(self.steps[-1].next_x - self.steps[-1].x)
        return last > first * 1.05

    @property
    def final_x(self) -> float:
        return self.steps[-1].next_x if self.steps else self.x0

    def path(self) -> list[tuple[float, float]]:
        return [(s.x, s.y) for s in self.steps]


def _finite(v: float) -> bool:
    return v == v and abs(v) != float("inf")


def minimum_of(f: Callable[[float], float], lo: float, hi: float,
               samples: int = 2000) -> tuple[float, float]:
    """The true minimum on an interval, by dense sampling.

    Used to mark the target the optimiser is heading toward — found
    independently of the descent, so the scene can honestly show whether
    gradient descent actually reached it.
    """
    best_x = lo
    best_y = f(lo)
    for i in range(1, samples + 1):
        x = lo + (hi - lo) * i / samples
        y = f(x)
        if y < best_y:
            best_x, best_y = x, y
    return best_x, best_y
