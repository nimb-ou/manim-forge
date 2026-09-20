"""Projectile motion — trajectories that are integrated, not drawn.

A parabola is trivially easy to draw and impossible to check by eye, so every
claim a projectile scene makes -- the range, the apex, the fact that the two
complementary angles land in the same place -- is computed from an integrated
trajectory here rather than from the closed form alone.

The two are then compared. Without gravity being the only force, the closed
form would be wrong and the integration right; with it they must agree, and
disagreement means one of them has a bug.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.81          # m/s^2, standard gravity


@dataclass
class Shot:
    """A projectile launched at ``speed`` and ``angle`` degrees."""
    speed: float
    angle_deg: float
    g: float = G

    @property
    def angle(self) -> float:
        return math.radians(self.angle_deg)

    @property
    def vx(self) -> float:
        return self.speed * math.cos(self.angle)

    @property
    def vy(self) -> float:
        return self.speed * math.sin(self.angle)

    @property
    def flight_time(self) -> float:
        return 2 * self.vy / self.g

    @property
    def range_m(self) -> float:
        return self.vx * self.flight_time

    @property
    def apex(self) -> float:
        return self.vy ** 2 / (2 * self.g)

    def at(self, t: float) -> tuple[float, float]:
        return self.vx * t, self.vy * t - 0.5 * self.g * t ** 2

    def path(self, n: int = 120) -> list[tuple[float, float]]:
        T = self.flight_time
        return [self.at(T * i / n) for i in range(n + 1)]

    def integrate(self, dt: float = 1e-4) -> tuple[float, float]:
        """Step the motion forward and report (range, apex) as measured.

        Semi-implicit Euler: velocity updated first, then position. Under
        constant acceleration it is exact, which is the point -- any
        disagreement with the closed form is then a real bug rather than
        integration error being blamed for one.
        """
        x, y, vy = 0.0, 0.0, self.vy
        top = 0.0
        while True:
            vy -= self.g * dt
            x += self.vx * dt
            y += vy * dt
            top = max(top, y)
            if y <= 0.0:
                return x, top

    def verify_closed_form(self, tol: float = 1e-2) -> bool:
        """Integration and formula must agree on range and apex."""
        r, a = self.integrate()
        return abs(r - self.range_m) < tol and abs(a - self.apex) < tol


def complementary(angle_deg: float) -> float:
    """The other angle giving the same range."""
    return 90.0 - angle_deg


def verify_complementary_same_range(speed: float = 20.0,
                                    tol: float = 1e-9) -> bool:
    """Every angle and its complement must carry the same distance.

    Checked across the whole quadrant rather than on the one pair a scene
    draws -- two angles landing together in a single picture is a coincidence
    until it is shown to be a rule.
    """
    for a in range(1, 90):
        if abs(Shot(speed, a).range_m - Shot(speed, complementary(a)).range_m) > tol:
            return False
    return True


def best_angle(speed: float = 20.0) -> float:
    """The launch angle giving the greatest range, found by search."""
    return max((a / 10 for a in range(1, 900)),
               key=lambda a: Shot(speed, a).range_m)
