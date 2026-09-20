"""Chemistry primitives — shells, bonds, and reaction kinetics.

Chemistry is the domain where a generated animation is most likely to be
confidently wrong, because its numbers look arbitrary. Two, eight, eighteen
reads as a list to memorise, so a model that has half-learned it will produce
two, eight, ten and nothing about the picture will look off.

So the numbers here are derived. Shell capacity is counted from the orbitals
that make it up rather than looked up, the bond length is found by minimising
a potential rather than placed, and the decay curve is checked against a
simulation of individual particles actually reacting.

Measured constants -- H2's bond length and dissociation energy -- are marked as
measured. They are data, not derivations, and pretending otherwise would be
the same dishonesty in the other direction.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------- shells


def orbitals_in_shell(n: int) -> list[tuple[int, int]]:
    """The (l, m) orbitals in principal shell ``n``.

    l runs 0..n-1 (s, p, d, f), and each l has 2l+1 orientations. The sum of
    the odd numbers 1, 3, 5, ... up to 2n-1 is n squared, which is where the
    square in 2n^2 comes from -- not from anywhere else.
    """
    return [(l, m) for l in range(n) for m in range(-l, l + 1)]


def shell_capacity(n: int) -> int:
    """Electrons that fit in shell ``n``: two per orbital, counted."""
    return 2 * len(orbitals_in_shell(n))


def verify_capacity_is_2n2(max_n: int = 6) -> bool:
    """Counting orbitals must agree with the formula the scene displays."""
    return all(shell_capacity(n) == 2 * n * n for n in range(1, max_n + 1))


#: How shells actually fill for the first twenty elements. The third shell
#: pauses at eight and the fourth starts, because 4s sits below 3d in energy
#: -- the reason potassium and calcium exist as they do. This is the ordering
#: taught as "2, 8, 8, 2", and it is a fact about energies, not about capacity.
FILL_ORDER = (2, 8, 8, 18)


def fill_shells(z: int, order: tuple[int, ...] = FILL_ORDER) -> list[int]:
    """Distribute ``z`` electrons into shells, filling each before the next."""
    left, out = z, []
    for cap in order:
        take = min(left, cap)
        out.append(take)
        left -= take
        if left <= 0:
            break
    if left > 0:
        raise ValueError(f"{z} electrons do not fit in {order}")
    return out


def valence_electrons(z: int) -> int:
    """Electrons in the outermost occupied shell."""
    return fill_shells(z)[-1]


# ---------------------------------------------------------------- bonding

#: Measured, not derived: H2's equilibrium bond length in picometres and its
#: dissociation energy in kJ/mol. Standard spectroscopic values.
H2_BOND_LENGTH_PM = 74.0
H2_DISSOCIATION_KJ = 436.0


@dataclass
class MorsePotential:
    """Energy of two bonded atoms as a function of their separation.

    V(r) = D_e * (1 - exp(-a(r - r_e)))^2 - D_e

    Chosen over a parabola because it has the right shape at both ends: it
    climbs steeply as the nuclei are pushed together and flattens to a
    dissociation limit as they are pulled apart. A parabola says pulling the
    atoms apart costs infinite energy, which would make the scene's whole
    point -- that a bond can break -- a lie.
    """
    depth: float = H2_DISSOCIATION_KJ      # D_e, kJ/mol
    width: float = 0.034                   # a, per pm
    r_e: float = H2_BOND_LENGTH_PM         # equilibrium separation, pm

    def energy(self, r: float) -> float:
        return self.depth * (1 - math.exp(-self.width * (r - self.r_e))) ** 2 - self.depth

    def minimum(self, lo: float = 30.0, hi: float = 300.0,
                steps: int = 20000) -> tuple[float, float]:
        """Find the lowest point by scanning, rather than assuming r_e.

        The scan is the check: if the parameters ever drift so that the
        analytic minimum is not where the curve actually bottoms out, the
        scene would draw the bond at one distance and label it another.
        """
        best_r = lo
        best_v = self.energy(lo)
        for i in range(steps + 1):
            r = lo + (hi - lo) * i / steps
            v = self.energy(r)
            if v < best_v:
                best_r, best_v = r, v
        return best_r, best_v

    def verify_minimum_at_re(self, tol_pm: float = 0.5) -> bool:
        return abs(self.minimum()[0] - self.r_e) < tol_pm


# ---------------------------------------------------------------- kinetics


@dataclass
class DecayRun:
    """One stochastic run of a first-order reaction."""
    counts: list[int]
    dt: float
    p: float

    @property
    def k(self) -> float:
        """Rate constant implied by the per-step reaction probability."""
        return -math.log(1 - self.p) / self.dt

    @property
    def half_life(self) -> float:
        return math.log(2) / self.k

    def predicted(self, t: float) -> float:
        return self.counts[0] * math.exp(-self.k * t)

    def times(self) -> list[float]:
        return [i * self.dt for i in range(len(self.counts))]


def simulate_decay(n0: int = 400, p: float = 0.06, steps: int = 60,
                   dt: float = 1.0, *, seed: int = 0) -> DecayRun:
    """Run the reaction particle by particle.

    Each surviving particle independently reacts with probability ``p`` per
    step. Nothing here assumes an exponential; the exponential is what a
    constant per-particle hazard produces, and the scene's claim is exactly
    that -- so it has to emerge, not be drawn.
    """
    rng = random.Random(seed)
    alive = n0
    counts = [alive]
    for _ in range(steps):
        alive -= sum(1 for _ in range(alive) if rng.random() < p)
        counts.append(alive)
    return DecayRun(counts=counts, dt=dt, p=p)


def verify_decay_matches_exponential(trials: int = 30, sigmas: float = 4.0) -> bool:
    """Averaged over runs, the simulation must track C0 * exp(-kt).

    Tested against the *sampling uncertainty*, not against a fixed percentage.
    Survivors at time t are binomial with q = exp(-kt), so the mean of
    ``trials`` runs has standard error sqrt(n0*q*(1-q)/trials). Deep in the
    tail that error is a large fraction of a small number: at q = 0.033 the
    mean of thirty runs is worth about 0.65 either way, so a 5% relative
    tolerance would demand one-standard-error accuracy and fail a third of the
    time on a perfectly correct model.

    A fixed tolerance here does not measure the model; it measures how far
    into the tail you looked.
    """
    runs = [simulate_decay(seed=s) for s in range(trials)]
    ref = runs[0]
    n0 = ref.counts[0]
    for i in range(0, len(ref.counts), 5):
        t = i * ref.dt
        mean = sum(r.counts[i] for r in runs) / trials
        q = math.exp(-ref.k * t)
        pred = n0 * q
        se = math.sqrt(n0 * q * (1 - q) / trials)
        if se == 0:                      # t = 0: every run starts identical
            if mean != pred:
                return False
            continue
        if abs(mean - pred) > sigmas * se:
            return False
    return True


def decay_residuals(trials: int = 30) -> list[tuple[float, float, float, float]]:
    """(t, observed mean, predicted, deviation in standard errors).

    Exposed so a scene -- or a reader doubting the check above -- can see the
    residuals rather than take a boolean on faith.
    """
    runs = [simulate_decay(seed=s) for s in range(trials)]
    ref = runs[0]
    n0 = ref.counts[0]
    out = []
    for i in range(0, len(ref.counts), 5):
        t = i * ref.dt
        mean = sum(r.counts[i] for r in runs) / trials
        q = math.exp(-ref.k * t)
        pred, se = n0 * q, math.sqrt(n0 * q * (1 - q) / trials)
        out.append((t, mean, pred, 0.0 if se == 0 else (mean - pred) / se))
    return out
