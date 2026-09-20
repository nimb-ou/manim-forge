"""Solid-geometry primitives — spheres, inscribed solids, and volumes.

Fourth entry in the verified-primitives library, same rule throughout: the
animation computes its own claims. A scene stating that the inscribed cube
fills 36.8% of its sphere must divide two volumes it calculated, and the cube
it draws must be the cube those volumes describe.

The inscribed-cube relation is the one worth getting right. A cube inside a
sphere touches it at its eight vertices, so the cube's *space diagonal* equals
the sphere's diameter — not its edge, and not its face diagonal. That gives
side = 2R/sqrt(3), and getting it wrong produces a cube that looks plausible
and is silently the wrong size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def sphere_volume(r: float) -> float:
    return 4.0 / 3.0 * math.pi * r ** 3


def cube_volume(side: float) -> float:
    return side ** 3


def inscribed_cube_side(r: float) -> float:
    """Edge of the largest cube fitting inside a sphere of radius ``r``.

    The cube meets the sphere at its eight vertices, so its space diagonal is
    the diameter: s*sqrt(3) = 2r.
    """
    return 2.0 * r / math.sqrt(3.0)


def circumscribed_cube_side(r: float) -> float:
    """Edge of the smallest cube containing the sphere — it touches at face
    centres, so the edge is simply the diameter. The contrast with the
    inscribed case is a good beat."""
    return 2.0 * r


@dataclass
class SphereCube:
    """A sphere with its largest inscribed cube, and the space between them."""
    radius: float = 1.0

    @property
    def side(self) -> float:
        return inscribed_cube_side(self.radius)

    @property
    def v_sphere(self) -> float:
        return sphere_volume(self.radius)

    @property
    def v_cube(self) -> float:
        return cube_volume(self.side)

    @property
    def v_gap(self) -> float:
        return self.v_sphere - self.v_cube

    @property
    def cube_fraction(self) -> float:
        return self.v_cube / self.v_sphere

    def vertices(self) -> list[tuple[float, float, float]]:
        h = self.side / 2.0
        return [(sx * h, sy * h, sz * h)
                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]

    def verify(self, tol: float = 1e-9) -> bool:
        """Every vertex must lie exactly on the sphere.

        This is the claim the scene rests on, so it is checked rather than
        trusted — a cube drawn slightly too large still looks like a cube
        inside a sphere, and nothing on screen would give it away.
        """
        return all(
            abs(math.sqrt(x * x + y * y + z * z) - self.radius) < tol
            for x, y, z in self.vertices()
        )


def fibonacci_sphere(n: int, radius: float = 1.0) -> list[tuple[float, float, float]]:
    """``n`` points spread evenly over a sphere, by the Fibonacci spiral.

    Even spacing matters visually: sampling latitude and longitude uniformly
    instead clusters points densely at the poles and sparsely at the equator,
    which reads as a mistake even to a viewer who cannot say why.
    """
    pts = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i / max(n - 1, 1)) * 2.0          # 1 down to -1
        r_at_y = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        pts.append((math.cos(theta) * r_at_y * radius,
                    y * radius,
                    math.sin(theta) * r_at_y * radius))
    return pts


@dataclass
class BallSample:
    """One uniformly-random point inside the ball, and where it landed."""
    x: float
    y: float
    z: float
    inside_cube: bool


def sample_ball(n: int, radius: float = 1.0, *, seed: int = 0) -> list[BallSample]:
    """``n`` points spread uniformly through the *volume* of the ball.

    Rejection sampling from the enclosing box: draw uniformly in
    [-R, R]^3 and keep the point only if it falls within the sphere. The
    survivors are exactly uniform over the ball, which the tempting shortcut
    -- uniform radius with a uniform direction -- is not; that one piles points
    towards the centre, because a shell at radius r has area growing as r^2.
    Acceptance rate here is the volume ratio pi/6, about 52%.

    Each point is tagged with whether it also lies inside the inscribed cube,
    so a scene can *count* the split rather than assert it.
    """
    import random

    rng = random.Random(seed)
    half = inscribed_cube_side(radius) / 2.0
    out: list[BallSample] = []
    while len(out) < n:
        x = rng.uniform(-radius, radius)
        y = rng.uniform(-radius, radius)
        z = rng.uniform(-radius, radius)
        if x * x + y * y + z * z > radius * radius:
            continue                      # outside the sphere: reject
        out.append(BallSample(
            x, y, z,
            inside_cube=max(abs(x), abs(y), abs(z)) <= half,
        ))
    return out


def counted_cube_fraction(samples: list[BallSample]) -> float:
    """The share of sampled points that landed inside the cube."""
    if not samples:
        return 0.0
    return sum(1 for s in samples if s.inside_cube) / len(samples)


def verify_sample_converges(radius: float = 1.0, n: int = 40000,
                            tol: float = 0.01) -> bool:
    """The counted fraction must approach the exact one.

    This is the check that keeps the Monte Carlo honest: if the sampler were
    biased towards the centre, the count would overstate the cube's share and
    the scene would display a wrong number with complete confidence.
    """
    exact = SphereCube(radius=radius).cube_fraction
    got = counted_cube_fraction(sample_ball(n, radius, seed=7))
    return abs(got - exact) < tol


def verify_sample_unbiased(radius: float = 1.0, n: int = 800, trials: int = 40,
                           tol_pp: float = 1.0) -> bool:
    """Averaged over independent seeds, the count must sit on the exact value.

    A single sample is allowed to miss -- that is what sampling error *is* --
    so checking one seed would either pass by luck or fail by luck. Averaging
    many independent seeds tests the thing that actually matters: that the
    estimator is centred on the truth rather than leaning one way.
    """
    exact = SphereCube(radius=radius).cube_fraction
    mean = sum(counted_cube_fraction(sample_ball(n, radius, seed=s))
               for s in range(trials)) / trials
    return abs(mean - exact) * 100.0 < tol_pp
