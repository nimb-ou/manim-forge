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


# ------------------------------------------------- Monty Hall and Bayes


@dataclass
class MontyRun:
    """One play: where the car was, what was picked, what was opened."""
    car: int
    first_pick: int
    opened: int
    switch_wins: bool
    stay_wins: bool


def play_monty(seed: int = 0, trials: int = 1000, doors: int = 3
               ) -> list[MontyRun]:
    """Play the door game properly, ``trials`` times.

    The host's constraint is the whole puzzle and it is enforced here: he
    opens a door that is neither the contestant's pick nor the car. A
    simulation that lets him open at random gives one half and quietly
    answers a different question -- which is why so many arguments about this
    puzzle are really arguments about which game is being played.
    """
    import random
    rng = random.Random(seed)
    out: list[MontyRun] = []
    for _ in range(trials):
        car = rng.randrange(doors)
        pick = rng.randrange(doors)
        choices = [d for d in range(doors) if d != pick and d != car]
        opened = rng.choice(choices)
        switched_to = next(d for d in range(doors)
                           if d != pick and d != opened)
        out.append(MontyRun(car, pick, opened,
                            switch_wins=(switched_to == car),
                            stay_wins=(pick == car)))
    return out


def monty_rates(runs: list[MontyRun]) -> tuple[float, float]:
    """(switch win rate, stay win rate) as actually observed."""
    n = len(runs)
    return (sum(r.switch_wins for r in runs) / n,
            sum(r.stay_wins for r in runs) / n)


def verify_monty_is_two_thirds(trials: int = 20000, tol: float = 0.02) -> bool:
    """Switching must win about two thirds, and the two rates must sum to one.

    The second half matters more than the first: switching and staying are
    complementary here, so any simulation where they do not sum to one has
    got the host's rule wrong.
    """
    sw, st = monty_rates(play_monty(seed=1, trials=trials))
    return abs(sw - 2 / 3) < tol and abs(sw + st - 1.0) < 1e-9


@dataclass
class TestOutcome:
    """A screening test applied to a whole population, in counts not rates."""
    population: int
    prevalence: float
    sensitivity: float
    specificity: float

    @property
    def sick(self) -> int:
        return round(self.population * self.prevalence)

    @property
    def well(self) -> int:
        return self.population - self.sick

    @property
    def true_positive(self) -> int:
        return round(self.sick * self.sensitivity)

    @property
    def false_positive(self) -> int:
        return round(self.well * (1 - self.specificity))

    @property
    def positive(self) -> int:
        return self.true_positive + self.false_positive

    @property
    def posterior(self) -> float:
        """Chance of being ill given a positive result."""
        return self.true_positive / self.positive if self.positive else 0.0

    def verify_against_bayes(self, tol: float = 1e-3) -> bool:
        """Counting people must agree with the formula.

        Counts are what the scene shows, because the formula is where the
        intuition goes wrong -- but if the two disagreed the scene would be
        showing one thing and naming another.
        """
        p_pos = (self.sensitivity * self.prevalence
                 + (1 - self.specificity) * (1 - self.prevalence))
        bayes = self.sensitivity * self.prevalence / p_pos
        return abs(self.posterior - bayes) < tol


# ------------------------------------------------- the central limit theorem


def sample_means(n_per_sample: int, n_samples: int, *, seed: int = 0,
                 kind: str = "uniform") -> list[float]:
    """Means of many small samples drawn from a decidedly non-normal source.

    ``kind`` picks the source distribution. "uniform" is flat, "skewed" is
    exponential-ish, "bimodal" has two humps and no mass in the middle. The
    point of offering ugly sources is that the theorem is about the *means*,
    not about the data -- a demonstration starting from a bell curve proves
    nothing at all.
    """
    import random
    rng = random.Random(seed)

    def draw() -> float:
        if kind == "uniform":
            return rng.random()
        if kind == "skewed":
            return rng.expovariate(1.6)
        if kind == "bimodal":
            return rng.gauss(0.18, 0.06) if rng.random() < 0.5 else rng.gauss(0.82, 0.06)
        raise ValueError(kind)

    return [sum(draw() for _ in range(n_per_sample)) / n_per_sample
            for _ in range(n_samples)]


def histogram(values: list[float], bins: int, lo: float, hi: float) -> list[int]:
    """Counts per bin, with out-of-range values clamped into the end bins."""
    out = [0] * bins
    for v in values:
        idx = int((v - lo) / (hi - lo) * bins)
        out[min(max(idx, 0), bins - 1)] += 1
    return out


def spread(values: list[float]) -> float:
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def verify_spread_shrinks_as_sqrt_n(kind: str = "bimodal",
                                    tol: float = 0.18) -> bool:
    """The spread of sample means must fall like one over root n.

    This is the quantitative half of the theorem and the half a picture
    cannot show: the histogram narrowing is obvious, but whether it narrows
    at the right *rate* is the actual claim. Checked by comparing the ratio
    of measured spreads against the ratio of one over root n.
    """
    base = spread(sample_means(1, 4000, seed=3, kind=kind))
    for n in (4, 16, 64):
        got = spread(sample_means(n, 4000, seed=3, kind=kind))
        want = base / math.sqrt(n)
        if abs(got - want) / want > tol:
            return False
    return True
