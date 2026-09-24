"""Forge kit: 3Blue1Brown-style building blocks a 7B model can call correctly.

The split's first long renders were slides: 215 of 289 beats built nothing
but text, for intents like "basis vectors i-hat and j-hat highlighted". A
7B model asked to invent an animation in raw Manim falls back on the one
thing it can always make work, a Text. This file gives it a vocabulary in
which the easy call *is* the picture: ``vector(stage, plane, (2, 1))`` draws
an arrow on a grid, ``riemann(stage, ax, g, 0, 2)`` refines rectangles under
a curve, ``apply_matrix(stage, plane, [[1, 1], [0, 1]], vecs)`` shears the
plane with its vectors riding along.

Rules every block follows, so a model cannot get them wrong:
  * it takes the ``stage`` first, plays its own animations, and returns the
    mobjects it made, for later beats to reuse;
  * it places what it makes in a region of the stage, scaled to fit, so
    nothing lands off screen or on top of the caption;
  * text goes through ``stage.title`` and ``stage.caption`` only, which
    replace what was there -- captions cannot pile up.

Self-contained on purpose (manim and numpy only): the assembler prepends
this file to a scene, so the same code renders on this Mac, on Kaggle for
the GRPO reward, and anywhere else, with no import path to get wrong.
"""
from __future__ import annotations

import numpy as np
from manim import (BLUE, BLUE_D, DOWN, GREEN, GREY, GREY_B, LEFT, ORIGIN,
                   ORANGE, PI, RED, RIGHT, TAU, TEAL, UP, WHITE, YELLOW,
                   AnimationGroup, ApplyMatrix, Arrow, Axes, Circle,
                   Circumscribe, Create, DashedLine, DecimalNumber, Dot,
                   FadeIn, FadeOut, GrowArrow, GrowFromCenter, Indicate,
                   Integer, LaggedStart, Line, MathTex, MoveAlongPath,
                   NumberLine, NumberPlane, Polygon, Rectangle, ReplacementTransform,
                   Sector, Square, Tex, Text, Transform, TransformMatchingTex,
                   TracedPath, ValueTracker, VGroup, Write, always_redraw,
                   config, ArrowVectorField, ComplexPlane, Rotate,
                   NumberPlane as _NP)

# -- the stage -----------------------------------------------------------------

_REGIONS = {
    # centre x, centre y, width, height, in frame units (14.2 x 8)
    "center": (0.0, -0.1, 9.0, 5.4),
    "left": (-3.4, -0.1, 6.2, 5.4),
    "right": (3.4, -0.1, 6.2, 5.4),
    "full": (0.0, -0.1, 13.0, 5.8),
}
_PALETTE = [BLUE, YELLOW, GREEN, RED, TEAL, ORANGE]


class Stage:
    """Owns the layout: a title slot, a caption slot, and regions between.

    ``stage.title("...")`` and ``stage.caption("...")`` replace what is in
    their slot; ``stage.place(m, "left")`` scales and moves a mobject into a
    region; ``stage.clear()`` fades out everything but the title.
    """

    def __init__(self, scene):
        self.scene = scene
        self._title = None
        self._caption = None
        self._objects: list = []

    # text slots ---------------------------------------------------------
    def _text(self, s: str, size: int):
        # A string with math in it becomes MathTex/Tex; plain words are Text.
        if "$" in s:
            return Tex(s, font_size=size)
        if any(c in s for c in "\\^_") and " " not in s.strip():
            return MathTex(s, font_size=size)
        return Text(s, font_size=size)

    def title(self, s: str, run_time: float = 1.0):
        new = self._text(s, 40).to_edge(UP, buff=0.35)
        new.scale_to_fit_width(min(new.width, 12.5))
        if self._title is None:
            self.scene.play(Write(new), run_time=run_time)
        else:
            # A cross-fade: morphing one title's letters into another's
            # reads as garble mid-transition.
            self.scene.play(FadeOut(self._title, shift=0.2 * UP),
                            FadeIn(new, shift=0.2 * UP), run_time=run_time)
        self._title = new
        return new

    def caption(self, s: str, run_time: float = 0.8):
        new = self._text(s, 30).to_edge(DOWN, buff=0.35)
        new.scale_to_fit_width(min(new.width, 12.5))
        if self._caption is None:
            self.scene.play(FadeIn(new, shift=0.2 * UP), run_time=run_time)
        else:
            self.scene.play(FadeOut(self._caption), FadeIn(new, shift=0.2 * UP),
                            run_time=run_time)
        self._caption = new
        return new

    def equation(self, *tex: str, where: str = "right", run_time: float = 1.2):
        """A derivation: each step transforms into the next, in a region."""
        cx, cy, w, h = _REGIONS[where]
        cur = MathTex(tex[0], font_size=40).move_to([cx, cy + h / 4, 0])
        cur.scale_to_fit_width(min(cur.width, w))
        self.scene.play(Write(cur), run_time=run_time)
        for t in tex[1:]:
            nxt = MathTex(t, font_size=40).move_to(cur)
            nxt.scale_to_fit_width(min(nxt.width, w))
            self.scene.play(TransformMatchingTex(cur, nxt), run_time=run_time)
            cur = nxt
        self._objects.append(cur)
        return cur

    # regions ------------------------------------------------------------
    def place(self, m, where: str = "center"):
        cx, cy, w, h = _REGIONS[where]
        if m.width > w or m.height > h:
            m.scale(min(w / max(m.width, 1e-6), h / max(m.height, 1e-6)))
        m.move_to([cx, cy, 0])
        self._objects.append(m)
        return m

    def label(self, m, s: str, direction=UP, color=WHITE):
        """A short label beside a mobject (not a caption)."""
        t = self._text(s, 28).set_color(color).next_to(m, direction, buff=0.15)
        self.scene.play(FadeIn(t), run_time=0.5)
        self._objects.append(t)
        return t

    def clear(self, keep=(), run_time: float = 0.8):
        keep_ids = {id(k) for k in keep}
        gone = [m for m in self.scene.mobjects
                if m is not self._title and id(m) not in keep_ids]
        if gone:
            self.scene.play(*[FadeOut(m) for m in gone], run_time=run_time)
        self._caption = None if self._caption in gone else self._caption
        self._objects = [m for m in self._objects if id(m) in keep_ids]

    def pause(self, seconds: float = 1.0):
        self.scene.wait(seconds)


def _need(obj, attr: str, what: str, call: str):
    """A clear error for the commonest misuse: the wrong thing in a slot.

    Without it, `vector(stage, (1, 0))` fails three frames deep in Manim with
    "'tuple' object has no attribute 'c2p'", which neither a model nor a
    repair prompt can act on.
    """
    if not hasattr(obj, attr):
        raise TypeError(f"{call}: expected {what}, got {type(obj).__name__} "
                        f"{obj!r:.40}")


# -- linear algebra ------------------------------------------------------------

def plane(stage: Stage, where: str = "center", x_extent: int = 4,
          y_extent: int = 3):
    """A faded coordinate grid filling its region, square units, drawn.

    Sized so a unit is as large as the region allows: at +-4 by +-3 in the
    centre region a unit is 0.9 frame units, and a (2, 1) vector reads from
    the back of the room. (A 10-by-8 grid scaled into the region made every
    vector a stub.)"""
    cx, cy, w, h = _REGIONS[where]
    unit = min(w / (2 * x_extent), h / (2 * y_extent))
    p = NumberPlane(x_range=[-x_extent, x_extent], y_range=[-y_extent, y_extent],
                    x_length=2 * x_extent * unit, y_length=2 * y_extent * unit,
                    background_line_style={"stroke_opacity": 0.45})
    stage.place(p, where)
    stage.scene.play(Create(p), run_time=1.5)
    return p


def vector(stage: Stage, plane_, xy, color=YELLOW, label: str | None = None):
    """An arrow from the plane's origin to (x, y), grown, optionally labelled."""
    _need(plane_, "c2p", "the plane from plane(stage)", "vector(stage, plane, (x, y))")
    a = Arrow(plane_.c2p(0, 0), plane_.c2p(*xy), buff=0, color=color)
    stage.scene.play(GrowArrow(a), run_time=0.8)
    stage._objects.append(a)
    if label:
        stage.label(a, label, direction=RIGHT if xy[0] >= 0 else LEFT,
                    color=color)
    return a


def basis(stage: Stage, plane_):
    """i-hat and j-hat, green and red, labelled."""
    _need(plane_, "c2p", "the plane from plane(stage)", "basis(stage, plane)")
    i = vector(stage, plane_, (1, 0), GREEN, r"\hat{\imath}")
    j = vector(stage, plane_, (0, 1), RED, r"\hat{\jmath}")
    return i, j


def apply_matrix(stage: Stage, plane_, matrix, riders=(), run_time: float = 2.0):
    """Move the grid, and anything riding on it, by a 2x2 matrix."""
    _need(plane_, "c2p", "the plane from plane(stage)", "apply_matrix(stage, plane, matrix)")
    m = np.array(matrix, dtype=float)
    about = plane_.c2p(0, 0)
    group = VGroup(plane_, *riders)
    stage.scene.play(ApplyMatrix(m, group, about_point=about), run_time=run_time)
    return group


def unit_square(stage: Stage, plane_, color=YELLOW):
    """The unit square on the grid, filled -- the area a determinant scales."""
    _need(plane_, "c2p", "the plane from plane(stage)", "unit_square(stage, plane)")
    sq = Polygon(plane_.c2p(0, 0), plane_.c2p(1, 0), plane_.c2p(1, 1),
                 plane_.c2p(0, 1), color=color, fill_opacity=0.35,
                 stroke_width=2)
    stage.scene.play(FadeIn(sq), run_time=0.8)
    stage._objects.append(sq)
    return sq


def span_line(stage: Stage, plane_, xy, color=BLUE):
    """Every scalar multiple of one vector: a line through the origin."""
    _need(plane_, "c2p", "the plane from plane(stage)", "span_line(stage, plane, (x, y))")
    d = np.array([*xy, 0.0]) / (np.linalg.norm(xy) or 1)
    ln = Line(plane_.c2p(*(-8 * d[:2])), plane_.c2p(*(8 * d[:2])),
              color=color, stroke_opacity=0.7)
    stage.scene.play(Create(ln), run_time=1.0)
    stage._objects.append(ln)
    return ln


def scale_vector(stage: Stage, plane_, arrow, factor: float, run_time=1.2):
    """Stretch an arrow along its own line by ``factor``."""
    stage.scene.play(arrow.animate.scale(factor, about_point=plane_.c2p(0, 0)),
                     run_time=run_time)
    return arrow


# -- functions and calculus ----------------------------------------------------

def axes(stage: Stage, x_range=(-1, 5), y_range=(-1, 5), where: str = "center",
         labels: tuple[str, str] = ("x", "y")):
    """Axes with tick numbers, drawn in a region."""
    ax = Axes(x_range=[*x_range, 1], y_range=[*y_range, 1],
              x_length=8, y_length=5, tips=False,
              axis_config={"include_numbers": True, "font_size": 24})
    stage.place(ax, where)
    lab = ax.get_axis_labels(MathTex(labels[0]), MathTex(labels[1]))
    stage.scene.play(Create(ax), FadeIn(lab), run_time=1.5)
    return ax


def graph(stage: Stage, ax, f, x_range=None, color=BLUE, label: str | None = None):
    """Plot f on the axes and draw it."""
    _need(ax, "plot", "the axes from axes(stage)", "graph(stage, axes, f)")
    xr = x_range or (ax.x_range[0], ax.x_range[1])
    g = ax.plot(f, x_range=[xr[0], xr[1]], color=color)
    stage.scene.play(Create(g), run_time=1.5)
    stage._objects.append(g)
    if label:
        t = MathTex(label, color=color, font_size=32).next_to(
            g.get_end(), UP + RIGHT, buff=0.1)
        stage.scene.play(FadeIn(t), run_time=0.5)
    return g


def tangent(stage: Stage, ax, f, x_start: float, x_end: float,
            color=YELLOW, run_time: float = 3.0):
    """A tangent line sliding along f, with its slope read out live."""
    _need(ax, "c2p", "the axes from axes(stage)", "tangent(stage, axes, f, x0, x1)")
    t = ValueTracker(x_start)
    h = 1e-4

    def line():
        x = t.get_value()
        k = (f(x + h) - f(x - h)) / (2 * h)
        p0 = ax.c2p(x - 1.2, f(x) - 1.2 * k)
        p1 = ax.c2p(x + 1.2, f(x) + 1.2 * k)
        return Line(p0, p1, color=color)

    tan = always_redraw(line)
    dot = always_redraw(lambda: Dot(ax.c2p(t.get_value(), f(t.get_value())),
                                    color=color))
    slope = DecimalNumber(0, num_decimal_places=2, font_size=34)
    slope.add_updater(lambda m: m.set_value(
        (f(t.get_value() + h) - f(t.get_value() - h)) / (2 * h)))
    read = VGroup(MathTex("\\text{slope} =", font_size=34), slope).arrange(RIGHT)
    read.next_to(ax, UP, buff=0.1).shift(3 * RIGHT)
    stage.scene.play(Create(tan), FadeIn(dot), FadeIn(read), run_time=1.0)
    stage.scene.play(t.animate.set_value(x_end), run_time=run_time)
    stage._objects += [tan, dot, read]
    return tan, dot, read


def area(stage: Stage, ax, g, a: float, b: float, color=BLUE_D):
    """Shade the area under g between a and b."""
    _need(ax, "get_area", "the axes from axes(stage)", "area(stage, axes, graph, a, b)")
    r = ax.get_area(g, x_range=[a, b], color=color, opacity=0.5)
    stage.scene.play(FadeIn(r), run_time=1.0)
    stage._objects.append(r)
    return r


def riemann(stage: Stage, ax, g, a: float, b: float, ns=(4, 8, 16, 32),
            color=TEAL):
    """Rectangles under g, refined: 4, 8, 16, 32 strips."""
    _need(ax, "get_riemann_rectangles", "the axes from axes(stage)", "riemann(stage, axes, graph, a, b)")
    rects = ax.get_riemann_rectangles(g, x_range=[a, b], dx=(b - a) / ns[0],
                                      fill_opacity=0.6, color=color)
    stage.scene.play(Create(rects), run_time=1.2)
    for n in ns[1:]:
        nxt = ax.get_riemann_rectangles(g, x_range=[a, b], dx=(b - a) / n,
                                        fill_opacity=0.6, color=color)
        stage.scene.play(Transform(rects, nxt), run_time=1.0)
    stage._objects.append(rects)
    return rects


def trace(stage: Stage, ax, f, x_start: float, x_end: float, color=YELLOW,
          run_time: float = 3.0):
    """A dot running along f and leaving a trail."""
    t = ValueTracker(x_start)
    d = always_redraw(lambda: Dot(ax.c2p(t.get_value(), f(t.get_value())),
                                  color=color))
    trail = TracedPath(d.get_center, stroke_color=color, stroke_width=3)
    stage.scene.add(trail)
    stage.scene.play(FadeIn(d), run_time=0.4)
    stage.scene.play(t.animate.set_value(x_end), run_time=run_time)
    stage._objects += [d, trail]
    return d, trail


# -- numbers and series --------------------------------------------------------

def number_line(stage: Stage, x_range=(0, 10), where: str = "center"):
    nl = NumberLine(x_range=[*x_range, 1], length=10, include_numbers=True)
    stage.place(nl, where)
    stage.scene.play(Create(nl), run_time=1.0)
    return nl


def point_on_line(stage: Stage, nl, x: float, color=YELLOW, label: str | None = None):
    d = Dot(nl.n2p(x), color=color)
    stage.scene.play(GrowFromCenter(d), run_time=0.5)
    stage._objects.append(d)
    if label:
        stage.label(d, label, color=color)
    return d


def bars(stage: Stage, values, labels=None, where: str = "center", color=BLUE,
         max_height: float = 4.0):
    """A bar chart, bars growing in one after another."""
    top = max(max(values), 1e-9)
    group = VGroup(*[Rectangle(width=0.6, height=max(v / top * max_height, 0.02),
                               fill_opacity=0.8, color=color, stroke_width=1)
                     for v in values]).arrange(RIGHT, buff=0.15, aligned_edge=DOWN)
    stage.place(group, where)
    stage.scene.play(LaggedStart(*[GrowFromCenter(b) for b in group],
                                 lag_ratio=0.15), run_time=1.5)
    if labels:
        tags = VGroup(*[MathTex(str(l), font_size=24).next_to(b, DOWN, buff=0.1)
                        for l, b in zip(labels, group)])
        stage.scene.play(FadeIn(tags), run_time=0.5)
        group.add(tags)
    return group


def partial_sums(stage: Stage, term, n: int = 10, where: str = "center",
                 color=YELLOW):
    """Bars of term(1)..term(n) stacking into a running total, read out."""
    total = 0.0
    sums = []
    for k in range(1, n + 1):
        total += term(k)
        sums.append(total)
    chart = bars(stage, sums, labels=list(range(1, n + 1)), where=where,
                 color=color)
    value = DecimalNumber(sums[-1], num_decimal_places=3, font_size=36)
    read = VGroup(MathTex("S_{%d} =" % n, font_size=36), value).arrange(RIGHT)
    read.next_to(chart, UP, buff=0.2)
    stage.scene.play(FadeIn(read), run_time=0.6)
    return chart, read


# -- geometry ------------------------------------------------------------------

def circle_slices(stage: Stage, n: int = 12, r: float = 1.6, where: str = "left"):
    """A circle cut into n sectors, alternately coloured."""
    cx, cy, _, _ = _REGIONS[where]
    secs = VGroup(*[Sector(radius=r, angle=TAU / n, start_angle=k * TAU / n,
                           fill_opacity=0.8, stroke_width=1,
                           color=BLUE if k % 2 == 0 else TEAL)
                    for k in range(n)]).move_to([cx, cy, 0])
    stage.scene.play(LaggedStart(*[FadeIn(s) for s in secs], lag_ratio=0.05),
                     run_time=1.2)
    stage._objects.append(secs)
    return secs


def unroll_to_rectangle(stage: Stage, secs, r: float = 1.6, where: str = "right"):
    """Lay the sectors alternately tip-down and tip-up into a near-rectangle
    of width pi*r and height r -- the picture behind area = pi r^2."""
    n = len(secs)
    ang = TAU / n
    cx, cy, _, _ = _REGIONS[where]
    w = 2 * r * np.sin(ang / 2)                  # chord of one wedge
    x0 = cx - (n / 2) * w / 2
    targets = []
    for k in range(n):
        up = k % 2 == 0                          # opens upward, tip at bottom
        t = Sector(radius=r, angle=ang,
                   start_angle=(PI / 2 - ang / 2) if up else (3 * PI / 2 - ang / 2),
                   fill_opacity=0.8, stroke_width=1,
                   color=BLUE if k % 2 == 0 else TEAL)
        tip = np.array([x0 + (k // 2) * w + (0 if up else w / 2),
                        cy - r / 2 if up else cy + r / 2, 0])
        t.shift(tip - t.get_arc_center())
        targets.append(t)
    stage.scene.play(*[Transform(s, t) for s, t in zip(secs, targets)],
                     run_time=2.5)
    return secs


def right_triangle(stage: Stage, a: float = 3, b: float = 2, where: str = "center",
                   labels=("a", "b", "c")):
    """A right triangle with its sides labelled."""
    cx, cy, _, _ = _REGIONS[where]
    p0, p1, p2 = (np.array([cx - a / 2, cy - b / 2, 0]),
                  np.array([cx + a / 2, cy - b / 2, 0]),
                  np.array([cx + a / 2, cy + b / 2, 0]))
    tri = Polygon(p0, p1, p2, color=WHITE, fill_opacity=0.15)
    corner = Square(0.25, color=GREY_B).move_to(p1 + np.array([-0.125, 0.125, 0]))
    stage.scene.play(Create(tri), FadeIn(corner), run_time=1.2)
    tags = VGroup(MathTex(labels[0]).next_to(Line(p0, p1), DOWN),
                  MathTex(labels[1]).next_to(Line(p1, p2), RIGHT),
                  MathTex(labels[2]).move_to((p0 + p2) / 2 + 0.35 * (UP + LEFT)))
    stage.scene.play(FadeIn(tags), run_time=0.6)
    grp = VGroup(tri, corner, tags)
    stage._objects.append(grp)
    return grp


# -- probability ---------------------------------------------------------------

def dice_grid(stage: Stage, where: str = "center", highlight_sum: int | None = None):
    """The 36 outcomes of two dice, with one sum highlighted."""
    cells = VGroup()
    for i in range(1, 7):
        for j in range(1, 7):
            sq = Square(0.62, stroke_width=1, color=GREY)
            sq.add(MathTex(f"{i},{j}", font_size=20))
            if highlight_sum and i + j == highlight_sum:
                sq.set_fill(YELLOW, opacity=0.5)
            cells.add(sq)
    cells.arrange_in_grid(6, 6, buff=0.05)
    stage.place(cells, where)
    stage.scene.play(LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.02),
                     run_time=1.5)
    return cells


# -- more linear algebra ------------------------------------------------------

def determinant(stage: Stage, plane_, matrix, run_time: float = 2.0):
    """The unit square rides a matrix; its new area is the determinant."""
    sq = unit_square(stage, plane_)
    m = np.array(matrix, dtype=float)
    apply_matrix(stage, plane_, m, riders=[sq], run_time=run_time)
    d = float(np.linalg.det(m))
    tag = MathTex(r"\text{area} = %s" % (f"{d:g}"), font_size=36)
    tag.next_to(sq, RIGHT, buff=0.2)
    stage.scene.play(FadeIn(tag), run_time=0.6)
    stage._objects.append(tag)
    return sq, tag


def eigenvectors(stage: Stage, plane_, matrix, run_time: float = 2.5):
    """Vectors on the eigen-directions stay on their lines as the plane moves;
    an ordinary vector is knocked off its line."""
    m = np.array(matrix, dtype=float)
    vals, vecs = np.linalg.eig(m)
    riders = []
    for k in range(2):
        if abs(np.imag(vals[k])) > 1e-9:
            continue
        v = np.real(vecs[:, k])
        riders.append(span_line(stage, plane_, tuple(v), color=YELLOW))
        riders.append(vector(stage, plane_, tuple(v), YELLOW))
    e0 = np.real(vecs[:, 0])
    # A test vector on neither eigen-line (2D cross product by hand: numpy 2
    # rejects np.cross on 2-vectors).
    off_xy = (1, 1) if abs(e0[0] * 1 - e0[1] * 1) > 0.1 else (1, -1)
    off = vector(stage, plane_, off_xy, RED)
    riders.append(off)
    apply_matrix(stage, plane_, m, riders=riders, run_time=run_time)
    return riders


# -- more calculus -------------------------------------------------------------

def taylor(stage: Stage, ax, f, a: float, terms, colors=None, run_time: float = 1.2):
    """Taylor polynomials about a, one more term each time, closing in on f.

    ``terms`` is a list of the derivatives' values at a: [f(a), f'(a), ...].
    """
    import math
    colors = colors or [RED, ORANGE, YELLOW, GREEN, TEAL, BLUE]
    xr = (ax.x_range[0], ax.x_range[1])
    ymin, ymax = ax.y_range[0], ax.y_range[1]

    def poly(n):
        return lambda x: float(np.clip(
            sum(terms[k] * (x - a) ** k / math.factorial(k) for k in range(n + 1)),
            ymin - 1, ymax + 1))

    cur = ax.plot(poly(0), x_range=[*xr], color=colors[0])
    stage.scene.play(Create(cur), run_time=run_time)
    for n in range(1, len(terms)):
        nxt = ax.plot(poly(n), x_range=[*xr], color=colors[n % len(colors)])
        stage.scene.play(Transform(cur, nxt), run_time=run_time)
    stage._objects.append(cur)
    return cur


def unit_circle_wave(stage: Stage, turns: float = 1.0, run_time: float = 4.0):
    """A radius turning on the unit circle, its height drawn out as a sine wave."""
    c = Circle(radius=1.3, color=GREY_B).move_to([-4.2, -0.1, 0])
    ax = Axes(x_range=[0, TAU * turns, PI / 2], y_range=[-1.2, 1.2, 1],
              x_length=7, y_length=2.6, tips=False).move_to([2.2, -0.1, 0])
    t = ValueTracker(0)
    tip = lambda: c.get_center() + 1.3 * np.array([np.cos(t.get_value()),
                                                   np.sin(t.get_value()), 0])
    radius = always_redraw(lambda: Line(c.get_center(), tip(), color=YELLOW))
    dot = always_redraw(lambda: Dot(tip(), color=YELLOW))
    wave = always_redraw(lambda: ax.plot(np.sin, x_range=[0, max(t.get_value(), 1e-3)],
                                         color=YELLOW))
    link = always_redraw(lambda: DashedLine(
        tip(), ax.c2p(t.get_value(), np.sin(t.get_value())), color=GREY))
    stage.scene.play(Create(c), Create(ax), run_time=1.0)
    stage.scene.add(radius, dot, wave, link)
    stage.scene.play(t.animate.set_value(TAU * turns), run_time=run_time,
                     rate_func=lambda x: x)
    stage._objects += [c, ax, radius, dot, wave, link]
    return c, ax, wave


def vector_field(stage: Stage, f, where: str = "center"):
    """Arrows for a 2D field f(x, y) -> (u, v), drawn in."""
    field = ArrowVectorField(lambda p: np.array([*f(p[0], p[1]), 0.0]),
                             x_range=[-5, 5, 1], y_range=[-3, 3, 1],
                             length_func=lambda n: 0.45 * np.tanh(n))
    stage.place(field, where)
    stage.scene.play(LaggedStart(*[GrowArrow(a) for a in field], lag_ratio=0.01),
                     run_time=1.5)
    return field


# -- complex numbers -----------------------------------------------------------

def complex_plane(stage: Stage, where: str = "center"):
    cp = ComplexPlane(x_range=[-4, 4], y_range=[-3, 3], x_length=7.2,
                      y_length=5.4,
                      background_line_style={"stroke_opacity": 0.45})
    cp.add_coordinates()
    stage.place(cp, where)
    stage.scene.play(Create(cp), run_time=1.2)
    return cp


def multiply_by(stage: Stage, cp, z: complex, points=(1 + 0j, 1j),
                run_time: float = 2.0):
    """Multiplying by z rotates by its angle and scales by its length: shown on
    arrows to the given points, which turn and stretch together."""
    arrows = [Arrow(cp.n2p(0), cp.n2p(w), buff=0, color=c)
              for w, c in zip(points, _PALETTE)]
    stage.scene.play(*[GrowArrow(a) for a in arrows], run_time=0.8)
    ang, r = np.angle(z), abs(z)
    stage.scene.play(*[a.animate.rotate(ang, about_point=cp.n2p(0))
                       .scale(r, about_point=cp.n2p(0)) for a in arrows],
                     run_time=run_time)
    stage._objects += arrows
    return arrows


# -- networks and graphs -------------------------------------------------------

def neural_net(stage: Stage, layers=(3, 4, 2), where: str = "center",
               pulse_through: bool = True):
    """Layers of neurons joined by weights; a signal pulses left to right."""
    cx, cy, w, h = _REGIONS[where]
    cols = []
    for k, n in enumerate(layers):
        x = cx - w / 2 + w * (k + 0.5) / len(layers)
        col = VGroup(*[Circle(radius=0.22, color=WHITE, stroke_width=2)
                       .move_to([x, cy + (h * 0.8) * (0.5 - (i + 0.5) / n), 0])
                       for i in range(n)])
        cols.append(col)
    edges = VGroup(*[Line(a.get_center(), b.get_center(), stroke_width=1,
                          stroke_opacity=0.5, color=GREY_B, buff=0.22)
                     for c1, c2 in zip(cols, cols[1:]) for a in c1 for b in c2])
    stage.scene.play(LaggedStart(*[FadeIn(c) for c in cols], lag_ratio=0.3),
                     Create(edges), run_time=1.8)
    if pulse_through:
        for col in cols:
            stage.scene.play(*[n.animate.set_fill(TEAL, opacity=0.8) for n in col],
                             run_time=0.4)
    net = VGroup(edges, *cols)
    stage._objects.append(net)
    return net


def network(stage: Stage, nodes, edges, where: str = "center"):
    """A graph: nodes {name: (x, y)} in [-1, 1]^2, edges [(a, b), ...]."""
    cx, cy, w, h = _REGIONS[where]
    pos = {k: np.array([cx + x * w * 0.42, cy + y * h * 0.4, 0])
           for k, (x, y) in nodes.items()}
    dots = {k: Dot(p, radius=0.12, color=BLUE) for k, p in pos.items()}
    tags = VGroup(*[MathTex(str(k), font_size=28).next_to(d, UP, buff=0.1)
                    for k, d in dots.items()])
    lines = VGroup(*[Line(pos[a], pos[b], color=GREY_B) for a, b in edges])
    stage.scene.play(Create(lines), *[GrowFromCenter(d) for d in dots.values()],
                     FadeIn(tags), run_time=1.5)
    g = VGroup(lines, *dots.values(), tags)
    stage._objects.append(g)
    return dots, lines


# -- sampling ------------------------------------------------------------------

def histogram_grows(stage: Stage, sampler, bins, n: int = 400, steps: int = 8,
                    where: str = "center", color=BLUE, seed: int = 0):
    """Draw samples in batches and watch the histogram take its shape.

    ``sampler(rng, k)`` returns k samples; ``bins`` is a list of edges.
    """
    rng = np.random.default_rng(seed)
    edges = np.array(bins, dtype=float)
    counts = np.zeros(len(edges) - 1)
    width = 8.0 / len(counts)
    cx, cy, w, h = _REGIONS[where]
    base = cy - h / 2 + 0.3
    bars_ = VGroup(*[Rectangle(width=width * 0.9, height=0.01, fill_opacity=0.8,
                               color=color, stroke_width=0)
                     .move_to([cx - 4 + width * (i + 0.5), base, 0], aligned_edge=DOWN)
                     for i in range(len(counts))])
    stage.scene.add(bars_)
    for _ in range(steps):
        counts += np.histogram(sampler(rng, n // steps), bins=edges)[0]
        top = max(counts.max(), 1)
        new = VGroup(*[Rectangle(width=width * 0.9, height=max(c / top * (h - 0.8), 0.01),
                                 fill_opacity=0.8, color=color, stroke_width=0)
                       .move_to([cx - 4 + width * (i + 0.5), base, 0], aligned_edge=DOWN)
                       for i, c in enumerate(counts)])
        stage.scene.play(Transform(bars_, new), run_time=0.5)
    stage._objects.append(bars_)
    return bars_


# -- emphasis ------------------------------------------------------------------

def highlight(stage: Stage, m, color=YELLOW):
    """Draw the eye to one thing."""
    stage.scene.play(Circumscribe(m, color=color), run_time=1.0)
    return m


def pulse(stage: Stage, m):
    stage.scene.play(Indicate(m), run_time=0.8)
    return m


KIT_API = """\
stage = Stage(self) already exists. Every block plays its own animation and
returns what it made; keep the return value to reuse it in later beats.
  stage.title(s)  stage.caption(s)   -- replace the title / caption slot
  stage.equation(tex1, tex2, ..., where="right") -- a derivation, each step
      transforming into the next
  stage.label(m, s)  stage.clear(keep=[...])  stage.pause(seconds)
  plane(stage, where="center", x_extent=4, y_extent=3) -> NumberPlane
  vector(stage, p, (x, y), color=YELLOW, label=None) -> Arrow
  basis(stage, p) -> (i_hat, j_hat)
  apply_matrix(stage, p, [[a, b], [c, d]], riders=[arrows...])
  unit_square(stage, p) -> Polygon      span_line(stage, p, (x, y)) -> Line
  scale_vector(stage, p, arrow, factor)
  axes(stage, x_range=(a, b), y_range=(c, d), where="center") -> Axes
  graph(stage, ax, f, color=BLUE, label=None) -> graph   (f is a lambda)
  tangent(stage, ax, f, x_start, x_end)   -- slides, slope read out live
  area(stage, ax, g, a, b)   riemann(stage, ax, g, a, b, ns=(4, 8, 16, 32))
  trace(stage, ax, f, x_start, x_end)   -- a dot runs along f
  number_line(stage, x_range=(a, b)) -> NumberLine
  point_on_line(stage, nl, x, label=None) -> Dot
  bars(stage, values, labels=None) -> bars
  partial_sums(stage, term, n=10) -> (bars, readout)  (term is a lambda k: ...)
  circle_slices(stage, n=12) -> sectors   unroll_to_rectangle(stage, sectors)
  right_triangle(stage, a=3, b=2) -> triangle
  dice_grid(stage, highlight_sum=None) -> cells
  determinant(stage, p, [[a, b], [c, d]]) -> (square, area_label)
  eigenvectors(stage, p, [[a, b], [c, d]])  -- eigen-directions stay put
  taylor(stage, ax, f, a, [f(a), f'(a), f''(a), ...])  -- polynomials close in
  unit_circle_wave(stage)   -- a turning radius draws a sine wave
  vector_field(stage, lambda x, y: (u, v))
  complex_plane(stage) -> cp    multiply_by(stage, cp, z)  -- rotate and scale
  neural_net(stage, layers=(3, 4, 2)) -> net
  network(stage, {"A": (x, y), ...}, [("A", "B"), ...]) -> (dots, lines)
  histogram_grows(stage, lambda rng, k: rng.normal(size=k), bins=[...])
  highlight(stage, m)   pulse(stage, m)
Regions: "center", "left", "right", "full". Text only through stage.title,
stage.caption, stage.label and stage.equation.
"""
