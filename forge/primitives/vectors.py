"""Vector primitives — projection, angle, and linear transformations.

Third entry in the verified-primitives library, same rule as the others: the
animation computes its own claims. A scene claiming "the dot product is 6"
must arrive at 6 by taking the dot product, and the projection it draws must
land where the projection actually lands.

That is not pedantry. Draw a projection by eye and it will be subtly wrong in
a way most viewers cannot name but can feel — and a model trained on scenes
whose geometry is decorative learns to produce decoration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, o: "Vec2") -> "Vec2":
        return Vec2(self.x + o.x, self.y + o.y)

    def __sub__(self, o: "Vec2") -> "Vec2":
        return Vec2(self.x - o.x, self.y - o.y)

    def __mul__(self, k: float) -> "Vec2":
        return Vec2(self.x * k, self.y * k)

    __rmul__ = __mul__

    @property
    def norm(self) -> float:
        return math.hypot(self.x, self.y)

    def dot(self, o: "Vec2") -> float:
        return self.x * o.x + self.y * o.y

    def cross(self, o: "Vec2") -> float:
        """The z-component of the 3D cross product — signed parallelogram area.

        Signed on purpose: the sign says which side ``o`` lies on, which is
        what makes it useful for orientation as well as area.
        """
        return self.x * o.y - self.y * o.x

    def angle_to(self, o: "Vec2") -> float:
        """Angle between, in radians, via atan2 of cross and dot.

        Not acos(dot / |a||b|): that loses precision badly for nearly parallel
        vectors, where floating point pushes the argument past 1 and the call
        raises. atan2 is stable across the whole range.
        """
        return abs(math.atan2(self.cross(o), self.dot(o)))

    def unit(self) -> "Vec2":
        n = self.norm
        return Vec2(0.0, 0.0) if n == 0 else Vec2(self.x / n, self.y / n)

    def projection_onto(self, o: "Vec2") -> "Vec2":
        """The component of this vector along ``o``.

        This is the vector the animation should draw — the shadow ``self``
        casts on ``o`` — and it is the geometric meaning of the dot product.
        """
        d = o.dot(o)
        if d == 0:
            return Vec2(0.0, 0.0)
        return o * (self.dot(o) / d)

    def scalar_projection(self, o: "Vec2") -> float:
        """Signed length of that shadow. Negative when the angle is obtuse."""
        n = o.norm
        return 0.0 if n == 0 else self.dot(o) / n

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def __repr__(self) -> str:
        return f"Vec2({self.x:g}, {self.y:g})"


def dot_identity_check(a: Vec2, b: Vec2, tol: float = 1e-9) -> bool:
    """Verify a·b == |a||b|cos(theta) for these two vectors.

    The identity the scene asserts, checked numerically before it is shown.
    A claim on screen that the code has not verified is a claim we are
    trusting rather than knowing.
    """
    lhs = a.dot(b)
    rhs = a.norm * b.norm * math.cos(a.angle_to(b))
    return abs(lhs - rhs) < tol * max(1.0, abs(lhs))
