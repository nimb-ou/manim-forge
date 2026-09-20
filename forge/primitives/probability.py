"""Probability primitives — trials, convergence, and exact expectations.

Eighth in the verified-primitives library, and the one where computing the
claim matters most. A scene about the law of large numbers that plots a
hand-drawn curve settling onto one half is a lie about randomness: real runs
wander, and the wandering is the point.

So trials are actually sampled, with a fixed seed so the scene is reproducible
without being fake, and the theoretical value is derived separately.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from fractions import Fraction


@dataclass
class CoinRun:
    """A real sequence of flips and its running proportion of heads.

    Seeded so renders are reproducible. That is not the same as staged: the
    sequence is genuinely sampled, and a different seed gives a different
    wander around the same limit.
    """
    n: int = 600
    p: float = 0.5
    seed: int = 7
    flips: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        self.flips = [1 if rng.random() < self.p else 0 for _ in range(self.n)]

    def running_proportion(self) -> list[tuple[int, float]]:
        out, heads = [], 0
        for i, f in enumerate(self.flips, 1):
            heads += f
            out.append((i, heads / i))
        return out

    def deviation_at(self, k: int) -> float:
        return abs(self.running_proportion()[k - 1][1] - self.p)

    def standard_error(self, k: int) -> float:
        """The 1/sqrt(n) envelope the wander should stay inside."""
        return math.sqrt(self.p * (1 - self.p) / k)


def dice_expectation(sides: int = 6) -> Fraction:
    """Exact expected value of a fair die — a Fraction, never 3.5000001."""
    return Fraction(sum(range(1, sides + 1)), sides)


def birthday_collision(n: int, days: int = 365) -> float:
    """Probability that two of n people share a birthday, computed exactly."""
    if n > days:
        return 1.0
    p_all_distinct = 1.0
    for i in range(n):
        p_all_distinct *= (days - i) / days
    return 1.0 - p_all_distinct


def smallest_n_above(threshold: float, days: int = 365) -> int:
    """Fewest people for a collision chance above ``threshold``.

    Searched rather than recalled: the famous "23" is then a result the scene
    derived, not a number it was told.
    """
    n = 1
    while birthday_collision(n, days) < threshold:
        n += 1
    return n
