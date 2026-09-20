"""Number-theory primitives — sieves, divisors, and the structure they expose.

Ninth in the verified-primitives library. The sieve returns the *order* in
which numbers are struck out, not just the primes that survive, because the
striking is what the animation is about: a scene that shows only the final
primes has skipped the algorithm entirely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Strike:
    """One crossing-out: which multiple, of which prime, in which pass."""
    value: int
    prime: int
    pass_index: int


def sieve(limit: int) -> tuple[list[int], list[Strike]]:
    """(primes, strikes) up to and including ``limit``.

    Multiples start at p² rather than 2p, which is the sieve's real
    optimisation: every smaller multiple of p already carries a smaller prime
    factor and was struck in an earlier pass. Animating from 2p would show a
    sieve doing redundant work.
    """
    alive = [True] * (limit + 1)
    alive[0] = alive[1] = False
    strikes: list[Strike] = []
    p = 2
    pass_index = 0
    while p * p <= limit:
        if alive[p]:
            for m in range(p * p, limit + 1, p):
                if alive[m]:
                    alive[m] = False
                    strikes.append(Strike(m, p, pass_index))
            pass_index += 1
        p += 1
    return [i for i, a in enumerate(alive) if a], strikes


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def prime_count(limit: int) -> int:
    return len(sieve(limit)[0])


def verify_sieve(limit: int) -> bool:
    """Cross-check the sieve against trial division.

    Two independent methods agreeing is what makes the scene's prime list a
    fact rather than an assertion.
    """
    primes, _ = sieve(limit)
    return primes == [n for n in range(2, limit + 1) if is_prime(n)]


@dataclass
class GcdStep:
    """One round of Euclid's algorithm, as a rectangle being trimmed."""
    a: int
    b: int
    q: int
    r: int


def euclid(a: int, b: int) -> list[GcdStep]:
    """Euclid's algorithm, recorded.

    Each step cuts as many b-by-b squares off an a-by-b rectangle as will fit
    and keeps the strip left over, which is what makes the geometric telling
    work. The last non-zero remainder is the gcd.
    """
    out = []
    while b:
        q, r = divmod(a, b)
        out.append(GcdStep(a, b, q, r))
        a, b = b, r
    return out


def verify_euclid(limit: int = 120) -> bool:
    """Euclid must agree with the divisor found by brute force, for every pair."""
    for a in range(1, limit):
        for b in range(1, limit):
            steps = euclid(a, b)
            got = steps[-1].b if steps else a
            want = max(d for d in range(1, min(a, b) + 1)
                       if a % d == 0 and b % d == 0)
            if got != want:
                return False
    return True


def sqrt2_contradiction(max_den: int = 2000) -> tuple[int, int, float]:
    """The closest fraction to root two with denominator up to ``max_den``.

    Returned so a scene can show the approximations getting better and never
    arriving. It is evidence, not proof -- the proof is the parity argument --
    but it is honest evidence, and a scene that shows a fraction "equal to"
    root two is showing something false.
    """
    best = (1, 1, abs(1.0 - math.sqrt(2)))
    for den in range(1, max_den + 1):
        num = round(math.sqrt(2) * den)
        err = abs(num / den - math.sqrt(2))
        if err < best[2]:
            best = (num, den, err)
    return best


def verify_no_exact_fraction(max_den: int = 3000) -> bool:
    """No fraction with denominator under the limit squares to exactly two.

    Checked in integers: n*n == 2*d*d is decidable, while (n/d)**2 == 2.0 is a
    floating-point coin toss that would silently "find" a solution.
    """
    return not any(n * n == 2 * d * d
                   for d in range(1, max_den + 1)
                   for n in (round(math.sqrt(2) * d),
                             round(math.sqrt(2) * d) + 1))


def parity_of_square(n: int) -> str:
    """Whether n^2 is even or odd -- the engine of the root-two proof."""
    return "even" if (n * n) % 2 == 0 else "odd"
