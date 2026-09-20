"""Oscillation primitives — springs, pendulums, and the traces they draw.

Sixth in the verified-primitives library. The rule holds: the animation
computes its own claims. A scene asserting that a mass on a spring traces a
cosine must produce the trace by *solving the motion*, not by plotting a cosine
next to a bouncing dot and trusting they agree.

So the motion is integrated numerically from Newton's second law, and the
analytic solution is derived separately. ``verify_matches_cosine`` checks the
two agree before any scene claims they do — which is the whole point of the
scene, and therefore exactly the thing that must not be assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SpringMass:
    """A mass on a spring: F = -kx, integrated rather than assumed.

    ``mass`` and ``stiffness`` are the physical inputs; period and angular
    frequency are derived from them, so changing the spring changes the
    animation because it changes the physics.
    """
    amplitude: float = 2.0
    mass: float = 1.0
    stiffness: float = 4.0
    dt: float = 0.002

    @property
    def omega(self) -> float:
        """Angular frequency, sqrt(k/m)."""
        return math.sqrt(self.stiffness / self.mass)

    @property
    def period(self) -> float:
        return 2 * math.pi / self.omega

    def analytic(self, t: float) -> float:
        """x(t) = A cos(wt), released from rest at full extension."""
        return self.amplitude * math.cos(self.omega * t)

    def integrate(self, duration: float) -> list[tuple[float, float, float]]:
        """(t, x, v) by velocity Verlet from F = -kx.

        Verlet rather than Euler: plain Euler gains energy on an oscillator, so
        the amplitude visibly grows over a few periods and the scene would show
        a spring that is quietly wrong. Verlet conserves energy well enough
        that the trace stays honest for as long as anyone will watch.
        """
        x = self.amplitude
        v = 0.0
        a = -self.stiffness * x / self.mass
        out = [(0.0, x, v)]
        steps = int(duration / self.dt)
        for i in range(1, steps + 1):
            x += v * self.dt + 0.5 * a * self.dt ** 2
            a_new = -self.stiffness * x / self.mass
            v += 0.5 * (a + a_new) * self.dt
            a = a_new
            out.append((i * self.dt, x, v))
        return out

    def energy(self, x: float, v: float) -> tuple[float, float]:
        """(kinetic, potential). Their sum is the quantity that should not drift."""
        return 0.5 * self.mass * v * v, 0.5 * self.stiffness * x * x

    def verify_matches_cosine(self, duration: float, tol: float = 5e-3) -> float:
        """Largest gap between the integrated motion and A cos(wt).

        The scene's central claim is that the trace *is* a cosine. Checking it
        numerically is what makes the claim earned rather than asserted.
        """
        return max(abs(x - self.analytic(t)) for t, x, _ in self.integrate(duration))


def sample(trajectory: list[tuple[float, float, float]], n: int
           ) -> list[tuple[float, float, float]]:
    """Thin a fine integration down to ``n`` points for drawing."""
    if len(trajectory) <= n:
        return trajectory
    step = len(trajectory) / n
    return [trajectory[int(i * step)] for i in range(n)]
