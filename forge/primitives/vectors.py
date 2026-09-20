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


@dataclass(frozen=True)
class Mat2:
    """A 2x2 matrix, as the linear map it represents."""
    a: float
    b: float
    c: float
    d: float

    def apply(self, v: Vec2) -> Vec2:
        return Vec2(self.a * v.x + self.b * v.y, self.c * v.x + self.d * v.y)

    @property
    def det(self) -> float:
        return self.a * self.d - self.b * self.c

    @property
    def columns(self) -> tuple[Vec2, Vec2]:
        """Where the two basis vectors land -- which is the whole matrix."""
        return Vec2(self.a, self.c), Vec2(self.b, self.d)

    def __matmul__(self, o: "Mat2") -> "Mat2":
        return Mat2(self.a * o.a + self.b * o.c, self.a * o.b + self.b * o.d,
                    self.c * o.a + self.d * o.c, self.c * o.b + self.d * o.d)


def verify_columns_are_images(m: Mat2, tol: float = 1e-12) -> bool:
    """The columns must equal the images of (1,0) and (0,1).

    True by construction, and asserted because it is the sentence the whole
    scene rests on: a matrix is not a grid of numbers with a rule attached,
    it is a record of where the basis vectors go.
    """
    e1, e2 = m.apply(Vec2(1.0, 0.0)), m.apply(Vec2(0.0, 1.0))
    c1, c2 = m.columns
    return (abs(e1.x - c1.x) < tol and abs(e1.y - c1.y) < tol
            and abs(e2.x - c2.x) < tol and abs(e2.y - c2.y) < tol)


def polygon_area(pts: list[Vec2]) -> float:
    """Signed area by the shoelace formula. Sign records orientation."""
    s = 0.0
    for p, q in zip(pts, pts[1:] + pts[:1]):
        s += p.x * q.y - q.x * p.y
    return s / 2.0


def verify_det_is_area_factor(m: Mat2, tol: float = 1e-9) -> bool:
    """Any shape's area must scale by exactly |det|, and flip sign with it.

    Measured on a real polygon rather than argued about: the unit square is
    mapped and its shoelace area compared. Using a non-square polygon as well
    guards against a coincidence that only holds for axis-aligned shapes.
    """
    for pts in ([Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1)],
                [Vec2(-0.4, -0.2), Vec2(1.3, 0.1), Vec2(0.9, 1.4), Vec2(-0.7, 0.8)]):
        before = polygon_area(pts)
        after = polygon_area([m.apply(p) for p in pts])
        if abs(after - before * m.det) > tol:
            return False
    return True


def eigen2(m: Mat2) -> list[tuple[float, Vec2]]:
    """Real eigenvalues and unit eigenvectors, or an empty list if none.

    Solved from the characteristic polynomial. An empty result is a real
    answer -- a rotation genuinely has no direction it leaves alone -- and a
    scene that always finds two is drawing something that is not there.
    """
    tr, det = m.a + m.d, m.det
    disc = tr * tr - 4 * det
    if disc < -1e-12:
        return []
    root = math.sqrt(max(disc, 0.0))
    out = []
    for lam in ((tr + root) / 2, (tr - root) / 2):
        # (A - lam I) v = 0 ; take a row that is not entirely zero.
        if abs(m.b) > 1e-12:
            v = Vec2(m.b, lam - m.a)
        elif abs(m.c) > 1e-12:
            v = Vec2(lam - m.d, m.c)
        else:
            v = Vec2(1.0, 0.0) if abs(lam - m.a) < 1e-12 else Vec2(0.0, 1.0)
        n = v.norm
        if n < 1e-12:
            continue
        out.append((lam, Vec2(v.x / n, v.y / n)))
    return out


def verify_eigen(m: Mat2, tol: float = 1e-9) -> bool:
    """Every returned pair must actually satisfy Av = lambda v."""
    for lam, v in eigen2(m):
        av = m.apply(v)
        if abs(av.x - lam * v.x) > tol or abs(av.y - lam * v.y) > tol:
            return False
    return True
