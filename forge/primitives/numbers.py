"""Number-theory primitives — sieves, divisors, and the structure they expose.

Ninth in the verified-primitives library. The sieve returns the *order* in
which numbers are struck out, not just the primes that survive, because the
striking is what the animation is about: a scene that shows only the final
primes has skipped the algorithm entirely.
"""

from __future__ import annotations

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
