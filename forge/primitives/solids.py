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
