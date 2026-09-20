"""Machine-learning primitives — surfaces, boundaries, and convolutions.

Every scene on this module shows something being learned, which is the single
easiest thing to fake in an animation: draw the answer, then draw a path that
arrives at it. So the paths here are produced by running the update rule, and
the rule is the plain one, written out.

Gradient descent in two dimensions is where the honest version earns its keep.
On a bowl every learning rate looks fine. On a valley -- steep across, shallow
along -- a rate that is stable in one direction oscillates in the other, and
that is the behaviour worth animating because it is the behaviour people
actually hit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Surface2D:
    """A quadratic bowl, possibly stretched: f(x, y) = a x^2 + b y^2."""
    a: float = 1.0
    b: float = 1.0

    def f(self, x: float, y: float) -> float:
        return self.a * x * x + self.b * y * y

    def grad(self, x: float, y: float) -> tuple[float, float]:
        return 2 * self.a * x, 2 * self.b * y

    @property
    def condition(self) -> float:
        """Ratio of the steep curvature to the shallow one.

        The number that decides whether a single learning rate can suit both
        directions at once. A bowl has one; this is why a bowl teaches nothing.
        """
        return max(self.a, self.b) / min(self.a, self.b)

    @property
    def stable_rate(self) -> float:
        """Largest step that does not diverge: 1 / max curvature."""
        return 1.0 / (2 * max(self.a, self.b))


@dataclass
class Descent2D:
    """Gradient descent on a 2-D surface, one real step at a time."""
    surface: Surface2D
    rate: float
    start: tuple[float, float] = (-1.6, 1.3)
    path: list[tuple[float, float]] = field(default_factory=list)

    def run(self, steps: int = 40) -> list[tuple[float, float]]:
        x, y = self.start
        self.path = [(x, y)]
        for _ in range(steps):
            gx, gy = self.surface.grad(x, y)
            x -= self.rate * gx
            y -= self.rate * gy
            if not (math.isfinite(x) and math.isfinite(y)) or abs(x) + abs(y) > 1e6:
                break
            self.path.append((x, y))
        return self.path

    @property
    def diverged(self) -> bool:
        """True when steps grow rather than shrink.

        Tested on step size, not on overflow: a run can be hopelessly
        divergent while every number in it is still finite, and waiting for
        an overflow would call that run converged.
        """
        if len(self.path) < 4:
            return True
        d = [math.dist(a, b) for a, b in zip(self.path, self.path[1:])]
        return d[-1] > d[0]

    @property
    def final_loss(self) -> float:
        return self.surface.f(*self.path[-1])


def verify_rate_threshold(surface: Surface2D) -> bool:
    """Steps below the stable rate must converge, and well above it must not.

    The threshold is a real property of the surface, so the scene's claim that
    a rate is 'too large' is checked against the surface rather than against
    how the picture happens to look.
    """
    t = surface.stable_rate
    below = Descent2D(surface, t * 0.5).run(60)
    above = Descent2D(surface, t * 2.5).run(60)
    return (not Descent2D(surface, t * 0.5, path=below).diverged
            and Descent2D(surface, t * 2.5, path=above).diverged)


# ------------------------------------------------------------ boundaries


@dataclass
class Perceptron:
    """A straight-line classifier trained by the perceptron rule."""
    w: list[float] = field(default_factory=lambda: [0.0, 0.0])
    b: float = 0.0
    history: list[tuple[list[float], float]] = field(default_factory=list)

    def predict(self, p: tuple[float, float]) -> int:
        return 1 if self.w[0] * p[0] + self.w[1] * p[1] + self.b > 0 else -1

    def fit(self, points: list[tuple[float, float]], labels: list[int],
            epochs: int = 30, rate: float = 0.08) -> int:
        """Train, recording the boundary after every correction.

        Returns the number of corrections made. The perceptron only moves when
        it is wrong, so a scene animating its history shows the boundary
        jumping on mistakes and sitting still otherwise -- which is the whole
        character of the algorithm and is lost if the frames are interpolated.
        """
        self.history = [(list(self.w), self.b)]
        fixes = 0
        for _ in range(epochs):
            errors = 0
            for p, y in zip(points, labels):
                if self.predict(p) != y:
                    self.w[0] += rate * y * p[0]
                    self.w[1] += rate * y * p[1]
                    self.b += rate * y
                    self.history.append((list(self.w), self.b))
                    errors += 1
                    fixes += 1
            if errors == 0:
                break
        return fixes

    def accuracy(self, points, labels) -> float:
        return sum(self.predict(p) == y
                   for p, y in zip(points, labels)) / len(points)


def blobs(n: int = 36, seed: int = 2, sep: float = 1.2, spread: float = 0.55
          ) -> tuple[list[tuple[float, float]], list[int]]:
    """Two labelled clouds. ``sep`` sets how far apart, ``spread`` how wide.

    Defaults give a set a line can separate, reached after six corrections --
    enough for the boundary to visibly hunt before it settles. Widen the
    spread and the clouds overlap, at which point the perceptron never
    settles at all, which is a fact about the algorithm worth showing rather
    than a parameter to avoid.
    """
    import random
    rng = random.Random(seed)
    pts, labs = [], []
    for i in range(n):
        y = 1 if i % 2 else -1
        cx, cy = (sep, sep * 0.6) if y == 1 else (-sep, -sep * 0.6)
        pts.append((cx + rng.gauss(0, spread), cy + rng.gauss(0, spread)))
        labs.append(y)
    return pts, labs


def is_separable(points, labels, epochs: int = 200, rate: float = 0.03) -> bool:
    """Whether the perceptron can reach every point correctly.

    The perceptron convergence theorem says it will, given enough passes, if
    and only if a separating line exists -- so failing to reach 100% after
    many epochs is evidence of overlap, not of bad luck.
    """
    p = Perceptron()
    p.fit(points, labels, epochs=epochs, rate=rate)
    return p.accuracy(points, labels) == 1.0


# ---------------------------------------------------------- convolution


def convolve1d(signal: list[float], kernel: list[float]) -> list[float]:
    """Slide ``kernel`` over ``signal``, edges held rather than zero-padded.

    Zero padding would make the ends dip towards zero and a scene would show
    an artefact of the padding as if it were an effect of the filter.
    """
    k, n = len(kernel), len(signal)
    half = k // 2
    out = []
    for i in range(n):
        s = 0.0
        for j, w in enumerate(kernel):
            idx = min(max(i + j - half, 0), n - 1)
            s += signal[idx] * w
        out.append(s)
    return out


def verify_blur_preserves_total(signal: list[float], kernel: list[float],
                                tol: float = 1e-6) -> bool:
    """A kernel summing to one must leave the signal's mean roughly alone."""
    if abs(sum(kernel) - 1.0) > 1e-9:
        return False
    a = sum(signal) / len(signal)
    b = sum(convolve1d(signal, kernel)) / len(signal)
    return abs(a - b) < max(tol, abs(a) * 0.05)
