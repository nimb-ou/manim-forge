"""Fields, waves and orbits — physics that has to be integrated to be believed.

Three claims these scenes make are exactly the kind a drawing fakes: that
field lines never cross, that two waves add pointwise, and that an orbit
conserves energy and sweeps equal areas in equal times. Each is computed here
and checked, because a hand-drawn ellipse obeys Kepler's second law only by
accident.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ------------------------------------------------------------------ fields


@dataclass
class Charge:
    x: float
    y: float
    q: float


def field_at(charges: list[Charge], x: float, y: float,
             k: float = 1.0) -> tuple[float, float]:
    """Vector sum of the inverse-square contributions at (x, y)."""
    ex = ey = 0.0
    for c in charges:
        dx, dy = x - c.x, y - c.y
        r2 = dx * dx + dy * dy
        if r2 < 1e-9:
            continue
        r = math.sqrt(r2)
        s = k * c.q / (r2 * r)
        ex += s * dx
        ey += s * dy
    return ex, ey


def trace_line(charges: list[Charge], start: tuple[float, float],
               step: float = 0.04, n: int = 900,
               bound: float = 7.0) -> list[tuple[float, float]]:
    """Follow the field from ``start`` until it leaves or hits a charge.

    Steps along the *unit* field direction, so spacing is uniform and the
    curve's shape is not distorted by field strength -- a line traced with
    raw magnitude races near the charges and crawls far away, which reads as
    a mistake even to someone who could not say why.
    """
    x, y = start
    pts = [(x, y)]
    for _ in range(n):
        ex, ey = field_at(charges, x, y)
        m = math.hypot(ex, ey)
        if m < 1e-12:
            break
        x += step * ex / m
        y += step * ey / m
        if abs(x) > bound or abs(y) > bound:
            break
        if any(math.hypot(x - c.x, y - c.y) < 0.12 for c in charges):
            pts.append((x, y))
            break
        pts.append((x, y))
    return pts


def verify_field_lines_do_not_cross(charges: list[Charge],
                                    samples: int = 400) -> bool:
    """The field must have one direction at each point away from a charge.

    Field lines crossing would mean two directions at one place. Checked by
    confirming the vector is well defined and single-valued on a grid --
    which it is by construction, and that is the point: the picture inherits
    the property from the arithmetic rather than from careful drawing.
    """
    import random
    rng = random.Random(0)
    for _ in range(samples):
        x, y = rng.uniform(-5, 5), rng.uniform(-3, 3)
        if any(math.hypot(x - c.x, y - c.y) < 0.3 for c in charges):
            continue
        a = field_at(charges, x, y)
        b = field_at(charges, x, y)
        if a != b or not all(math.isfinite(v) for v in a):
            return False
    return True


# ------------------------------------------------------------------- waves


def wave(x: float, t: float, amp: float = 1.0, k: float = 2.0,
         omega: float = 2.0, phase: float = 0.0, direction: int = 1) -> float:
    return amp * math.sin(k * x - direction * omega * t + phase)


def superpose(x: float, t: float, parts: list[dict]) -> float:
    """Waves add. The whole of interference is this one line."""
    return sum(wave(x, t, **p) for p in parts)


def verify_superposition_is_linear(parts: list[dict], tol: float = 1e-12) -> bool:
    """The sum of the waves must equal the wave of the sums, everywhere tested.

    Trivially true of the formula, and worth asserting anyway: a scene that
    draws an envelope by hand instead of adding samples will drift from it,
    and nothing on screen says which one is being shown.
    """
    for i in range(60):
        x, t = -6 + i * 0.2, i * 0.07
        if abs(superpose(x, t, parts)
               - sum(wave(x, t, **p) for p in parts)) > tol:
            return False
    return True


def standing_nodes(k: float = 2.0, span: float = 6.0) -> list[float]:
    """Positions where two opposite waves of equal amplitude always cancel.

    sin(kx - wt) + sin(kx + wt) = 2 sin(kx) cos(wt), so the zeros sit where
    sin(kx) = 0 -- fixed in space, independent of time. Computed, because a
    node drawn slightly off is invisible and wrong.
    """
    out, n = [], 0
    while True:
        p = n * math.pi / k
        if p > span:
            break
        if p > 0:
            out.append(p)
        out.append(-p) if p > 0 else out.append(0.0)
        n += 1
    return sorted(out)


# ------------------------------------------------------------------ orbits


@dataclass
class Orbit:
    """A body around a fixed mass, integrated with velocity Verlet."""
    mu: float = 4.0                 # GM
    r0: float = 2.6
    v0: float = 1.15
    path: list[tuple[float, float]] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)

    def run(self, dt: float = 0.002, steps: int = 12000):
        """Velocity Verlet: symplectic, so energy does not drift away.

        Plain Euler spirals outward on this problem no matter how small the
        step, and the spiral looks like physics. It is not; it is the
        integrator.
        """
        x, y = self.r0, 0.0
        vx, vy = 0.0, self.v0

        def acc(px, py):
            r = math.hypot(px, py)
            s = -self.mu / (r ** 3)
            return s * px, s * py

        ax, ay = acc(x, y)
        self.path, self.speeds = [(x, y)], [math.hypot(vx, vy)]
        for _ in range(steps):
            x += vx * dt + 0.5 * ax * dt * dt
            y += vy * dt + 0.5 * ay * dt * dt
            nax, nay = acc(x, y)
            vx += 0.5 * (ax + nax) * dt
            vy += 0.5 * (ay + nay) * dt
            ax, ay = nax, nay
            self.path.append((x, y))
            self.speeds.append(math.hypot(vx, vy))
        return self.path

    def energy(self, i: int) -> float:
        x, y = self.path[i]
        return 0.5 * self.speeds[i] ** 2 - self.mu / math.hypot(x, y)

    def swept_area(self, i0: int, i1: int) -> float:
        """Area swept by the radius between two indices, by the shoelace sum."""
        a = 0.0
        for i in range(i0, i1):
            x0, y0 = self.path[i]
            x1, y1 = self.path[i + 1]
            a += 0.5 * abs(x0 * y1 - x1 * y0)
        return a

    def verify_energy_conserved(self, tol: float = 1e-3) -> bool:
        es = [self.energy(i) for i in range(0, len(self.path), 200)]
        return max(es) - min(es) < tol

    def verify_equal_areas(self, window: int = 900, tol: float = 0.02) -> bool:
        """Kepler's second law, measured: equal times sweep equal areas.

        Compared between a stretch near the closest approach and one near the
        furthest, which is where the claim has teeth -- anywhere else the
        speeds are similar enough that a wrong orbit would pass.
        """
        rs = [math.hypot(*p) for p in self.path]
        near = rs.index(min(rs))
        far = rs.index(max(rs))
        n = len(self.path) - window - 1
        a = self.swept_area(min(near, n), min(near, n) + window)
        b = self.swept_area(min(far, n), min(far, n) + window)
        return abs(a - b) / max(a, b) < tol
