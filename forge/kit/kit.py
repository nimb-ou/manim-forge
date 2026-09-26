"""Forge kit: 3Blue1Brown-style building blocks a 7B model can call correctly.

The split's first long renders were slides: 215 of 289 beats built nothing
but text, for intents like "basis vectors i-hat and j-hat highlighted". A
7B model asked to invent an animation in raw Manim falls back on the one
thing it can always make work, a Text. This file gives it a vocabulary in
which the easy call *is* the picture: ``draw_vector(stage, plane, (2, 1))`` draws
an arrow on a grid, ``riemann_refine(stage, ax, g, 0, 2)`` refines rectangles under
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

# -- 2D points, accepted ----------------------------------------------------
# Models write .shift((1, 2)) and .move_to((x, y)); Manim wants 3D points and
# fails deep inside with "operands could not be broadcast together with
# shapes (32,3) (2,)" -- the commonest runtime error in teacher data, and the
# first thing that broke in the app. Pad a 2-vector with z = 0.
from manim import Mobject as _Mobject


def _as_point(v):
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return v
    return np.append(a, 0.0) if a.shape == (2,) else v


def _pad_points(method_name: str, first_only: bool = False):
    original = getattr(_Mobject, method_name)

    def wrapped(self, *args, **kw):
        if args:
            args = ((_as_point(args[0]),) + args[1:]) if first_only \
                else tuple(_as_point(a) for a in args)
        return original(self, *args, **kw)
    wrapped.__name__ = original.__name__
    wrapped.__doc__ = original.__doc__
    setattr(_Mobject, method_name, wrapped)


if not getattr(_Mobject, "_forge_2d_ok", False):
    _pad_points("shift")
    _pad_points("move_to", first_only=True)
    _pad_points("next_to", first_only=True)
    _Mobject._forge_2d_ok = True


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
        # A caption is a phrase, not the narration. The first kit coder put
        # whole narration sentences here, shrunk to fit the width until they
        # were unreadable; the first clause, at most ten words, is kept.
        words = str(s).replace("\n", " ").split()
        if len(words) > 10:
            s = " ".join(words[:10])
            for stop in (". ", "; ", ", "):
                if stop in s:
                    s = s.split(stop)[0]
                    break
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

    def label(self, m, s: str, direction=UP, color=WHITE, **_ignored):
        """A short label beside a mobject (not a caption)."""
        if isinstance(m, str) and not isinstance(s, str):
            m, s = s, m                                  # label("x", obj)
        if isinstance(m, str):                           # label("x") alone
            return self.caption(m)
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

    # The obvious phrasings, accepted. The teacher's commonest runtime error
    # across 438 scenes was stage.play / stage.add (23), then a block called
    # as a method -- stage.draw_plane(...). The stage forwards the first to
    # the scene and binds the second, rather than failing the beat.
    def play(self, *anims, **kw):
        return self.scene.play(*anims, **kw)

    def add(self, *mobjects):
        return self.scene.add(*mobjects)

    def remove(self, *mobjects):
        return self.scene.remove(*mobjects)

    def wait(self, seconds: float = 1.0):
        return self.scene.wait(seconds)

    def __getattr__(self, name: str):
        f = globals().get(name)
        if callable(f) and getattr(f, "__code__", None) is not None \
                and f.__code__.co_varnames[:1] == ("stage",):
            return lambda *a, **kw: f(self, *a, **kw)
        raise AttributeError(
            f"Stage has no {name!r}. Blocks are functions taking the stage "
            f"first (draw_plane(stage), plot_graph(stage, ax, f)); the stage's "
            f"own methods are title, caption, equation, label, place, clear, "
            f"pause.")


def _need(obj, attr: str, what: str, call: str):
    """A clear error for the commonest misuse: the wrong thing in a slot.

    Without it, `draw_vector(stage, (1, 0))` fails three frames deep in Manim with
    "'tuple' object has no attribute 'c2p'", which neither a model nor a
    repair prompt can act on.
    """
    if not hasattr(obj, attr):
        raise TypeError(f"{call}: expected {what}, got {type(obj).__name__} "
                        f"{obj!r:.40}")


# -- linear algebra ------------------------------------------------------------

def draw_plane(stage: Stage, where: str = "center", x_extent: int = 4,
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


def draw_vector(stage: Stage, plane_, xy, color=YELLOW, label: str | None = None):
    """An arrow from the plane's origin to (x, y), grown, optionally labelled."""
    _need(plane_, "c2p", "the plane from draw_plane(stage)", "draw_vector(stage, plane, (x, y))")
    a = Arrow(plane_.c2p(0, 0), plane_.c2p(*xy), buff=0, color=color)
    stage.scene.play(GrowArrow(a), run_time=0.8)
    stage._objects.append(a)
    if label:
        stage.label(a, label, direction=RIGHT if xy[0] >= 0 else LEFT,
                    color=color)
    return a


def draw_basis(stage: Stage, plane_):
    """i-hat and j-hat, green and red, labelled."""
    _need(plane_, "c2p", "the plane from draw_plane(stage)", "draw_basis(stage, plane)")
    i = draw_vector(stage, plane_, (1, 0), GREEN, r"\hat{\imath}")
    j = draw_vector(stage, plane_, (0, 1), RED, r"\hat{\jmath}")
    return i, j


def apply_matrix(stage: Stage, plane_, matrix, riders=(), run_time: float = 2.0):
    """Move the grid, and anything riding on it, by a 2x2 matrix."""
    _need(plane_, "c2p", "the plane from draw_plane(stage)", "apply_matrix(stage, plane, matrix)")
    m = np.array(matrix, dtype=float)
    about = plane_.c2p(0, 0)
    group = VGroup(plane_, *riders)
    stage.scene.play(ApplyMatrix(m, group, about_point=about), run_time=run_time)
    return group


def draw_unit_square(stage: Stage, plane_, color=YELLOW):
    """The unit square on the grid, filled -- the area a determinant scales."""
    _need(plane_, "c2p", "the plane from draw_plane(stage)", "draw_unit_square(stage, plane)")
    sq = Polygon(plane_.c2p(0, 0), plane_.c2p(1, 0), plane_.c2p(1, 1),
                 plane_.c2p(0, 1), color=color, fill_opacity=0.35,
                 stroke_width=2)
    stage.scene.play(FadeIn(sq), run_time=0.8)
    stage._objects.append(sq)
    return sq


def draw_span(stage: Stage, plane_, xy, color=BLUE):
    """Every scalar multiple of one vector: a line through the origin."""
    _need(plane_, "c2p", "the plane from draw_plane(stage)", "draw_span(stage, plane, (x, y))")
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

def draw_axes(stage: Stage, x_range=(-1, 5), y_range=(-1, 5), where: str = "center",
         labels: tuple[str, str] = ("x", "y")):
    """Axes with tick numbers, drawn in a region."""
    ax = Axes(x_range=[*x_range, 1], y_range=[*y_range, 1],
              x_length=8, y_length=5, tips=False,
              axis_config={"include_numbers": True, "font_size": 24})
    stage.place(ax, where)
    lab = ax.get_axis_labels(MathTex(labels[0]), MathTex(labels[1]))
    stage.scene.play(Create(ax), FadeIn(lab), run_time=1.5)
    return ax


def plot_graph(stage: Stage, ax, f, x_range=None, color=BLUE, label: str | None = None):
    """Plot f on the axes and draw it."""
    _need(ax, "plot", "the axes from draw_axes(stage)", "plot_graph(stage, axes, f)")
    xr = x_range or (ax.x_range[0], ax.x_range[1])
    g = ax.plot(f, x_range=[xr[0], xr[1]], color=color)
    stage.scene.play(Create(g), run_time=1.5)
    stage._objects.append(g)
    if label:
        t = MathTex(label, color=color, font_size=32).next_to(
            g.get_end(), UP + RIGHT, buff=0.1)
        stage.scene.play(FadeIn(t), run_time=0.5)
    return g


def slide_tangent(stage: Stage, ax, f, x_start: float, x_end: float,
            color=YELLOW, run_time: float = 3.0):
    """A tangent line sliding along f, with its slope read out live."""
    _need(ax, "c2p", "the axes from draw_axes(stage)", "slide_tangent(stage, axes, f, x0, x1)")
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


def shade_area(stage: Stage, ax, g, a: float, b: float, color=BLUE_D):
    """Shade the area under g between a and b."""
    _need(ax, "get_area", "the axes from draw_axes(stage)", "shade_area(stage, axes, graph, a, b)")
    r = ax.get_area(g, x_range=[a, b], color=color, opacity=0.5)
    stage.scene.play(FadeIn(r), run_time=1.0)
    stage._objects.append(r)
    return r


def riemann_refine(stage: Stage, ax, g, a: float, b: float, ns=(4, 8, 16, 32),
            color=TEAL):
    """Rectangles under g, refined: 4, 8, 16, 32 strips."""
    _need(ax, "get_riemann_rectangles", "the axes from draw_axes(stage)", "riemann_refine(stage, axes, graph, a, b)")
    rects = ax.get_riemann_rectangles(g, x_range=[a, b], dx=(b - a) / ns[0],
                                      fill_opacity=0.6, color=color)
    stage.scene.play(Create(rects), run_time=1.2)
    for n in ns[1:]:
        nxt = ax.get_riemann_rectangles(g, x_range=[a, b], dx=(b - a) / n,
                                        fill_opacity=0.6, color=color)
        stage.scene.play(Transform(rects, nxt), run_time=1.0)
    stage._objects.append(rects)
    return rects


def trace_graph(stage: Stage, ax, f, x_start: float, x_end: float, color=YELLOW,
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

def draw_number_line(stage: Stage, x_range=(0, 10), where: str = "center"):
    """A number line with its integers labelled, drawn."""
    nl = NumberLine(x_range=[*x_range, 1], length=10, include_numbers=True)
    stage.place(nl, where)
    stage.scene.play(Create(nl), run_time=1.0)
    return nl


def mark_point(stage: Stage, nl, x: float, color=YELLOW, label: str | None = None):
    """A dot at x on a number line, optionally labelled."""
    d = Dot(nl.n2p(x), color=color)
    stage.scene.play(GrowFromCenter(d), run_time=0.5)
    stage._objects.append(d)
    if label:
        stage.label(d, label, color=color)
    return d


def draw_bars(stage: Stage, values, labels=None, where: str = "center", color=BLUE,
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


def show_partial_sums(stage: Stage, term, n: int = 10, where: str = "center",
                 color=YELLOW):
    """Bars of term(1)..term(n) stacking into a running total, read out."""
    total = 0.0
    sums = []
    for k in range(1, n + 1):
        total += term(k)
        sums.append(total)
    chart = draw_bars(stage, sums, labels=list(range(1, n + 1)), where=where,
                 color=color)
    value = DecimalNumber(sums[-1], num_decimal_places=3, font_size=36)
    read = VGroup(MathTex("S_{%d} =" % n, font_size=36), value).arrange(RIGHT)
    read.next_to(chart, UP, buff=0.2)
    stage.scene.play(FadeIn(read), run_time=0.6)
    return chart, read


# -- geometry ------------------------------------------------------------------

def slice_circle(stage: Stage, n: int = 12, r: float = 1.6, where: str = "left"):
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


def unroll_slices(stage: Stage, secs, r: float = 1.6, where: str = "right"):
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


def draw_right_triangle(stage: Stage, a: float = 3, b: float = 2, where: str = "center",
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

def draw_dice_grid(stage: Stage, where: str = "center", highlight_sum: int | None = None):
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

def show_determinant(stage: Stage, plane_, matrix, run_time: float = 2.0):
    """The unit square rides a matrix; its new area is the determinant."""
    sq = draw_unit_square(stage, plane_)
    m = np.array(matrix, dtype=float)
    apply_matrix(stage, plane_, m, riders=[sq], run_time=run_time)
    d = float(np.linalg.det(m))
    tag = MathTex(r"\text{area} = %s" % (f"{d:g}"), font_size=36)
    tag.next_to(sq, RIGHT, buff=0.2)
    stage.scene.play(FadeIn(tag), run_time=0.6)
    stage._objects.append(tag)
    return sq, tag


def show_eigenvectors(stage: Stage, plane_, matrix, run_time: float = 2.5):
    """Vectors on the eigen-directions stay on their lines as the plane moves;
    an ordinary vector is knocked off its line."""
    m = np.array(matrix, dtype=float)
    vals, vecs = np.linalg.eig(m)
    riders = []
    for k in range(2):
        if abs(np.imag(vals[k])) > 1e-9:
            continue
        v = np.real(vecs[:, k])
        riders.append(draw_span(stage, plane_, tuple(v), color=YELLOW))
        riders.append(draw_vector(stage, plane_, tuple(v), YELLOW))
    e0 = np.real(vecs[:, 0])
    # A test vector on neither eigen-line (2D cross product by hand: numpy 2
    # rejects np.cross on 2-vectors).
    off_xy = (1, 1) if abs(e0[0] * 1 - e0[1] * 1) > 0.1 else (1, -1)
    off = draw_vector(stage, plane_, off_xy, RED)
    riders.append(off)
    apply_matrix(stage, plane_, m, riders=riders, run_time=run_time)
    return riders


# -- more calculus -------------------------------------------------------------

def taylor_approximate(stage: Stage, ax, f, a: float, terms, colors=None, run_time: float = 1.2):
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


def circle_to_sine(stage: Stage, turns: float = 1.0, run_time: float = 4.0):
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


def draw_vector_field(stage: Stage, f, where: str = "center"):
    """Arrows for a 2D field f(x, y) -> (u, v), drawn in."""
    field = ArrowVectorField(lambda p: np.array([*f(p[0], p[1]), 0.0]),
                             x_range=[-5, 5, 1], y_range=[-3, 3, 1],
                             length_func=lambda n: 0.45 * np.tanh(n))
    stage.place(field, where)
    stage.scene.play(LaggedStart(*[GrowArrow(a) for a in field], lag_ratio=0.01),
                     run_time=1.5)
    return field


# -- complex numbers -----------------------------------------------------------

def draw_complex_plane(stage: Stage, where: str = "center"):
    """The complex plane with its coordinates, drawn."""
    cp = ComplexPlane(x_range=[-4, 4], y_range=[-3, 3], x_length=7.2,
                      y_length=5.4,
                      background_line_style={"stroke_opacity": 0.45})
    cp.add_coordinates()
    stage.place(cp, where)
    stage.scene.play(Create(cp), run_time=1.2)
    return cp


def multiply_complex(stage: Stage, cp, z: complex, points=(1 + 0j, 1j),
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

def draw_neural_net(stage: Stage, layers=(3, 4, 2), where: str = "center",
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


def draw_network(stage: Stage, nodes, edges, where: str = "center"):
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

def grow_histogram(stage: Stage, sampler, bins, n: int = 400, steps: int = 8,
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


# -- waves and series ----------------------------------------------------------

def animate_wave(stage: Stage, ax, amp: float = 1.0, k: float = 2.0, omega: float = 2.0,
         t_end: float = 4.0, color=YELLOW):
    """A travelling wave amp*sin(kx - wt), moving for t_end seconds."""
    _need(ax, "plot", "the axes from draw_axes(stage)", "animate_wave(stage, axes)")
    t = ValueTracker(0)
    xr = [ax.x_range[0], ax.x_range[1]]
    w = always_redraw(lambda: ax.plot(
        lambda x: amp * np.sin(k * x - omega * t.get_value()), x_range=xr,
        color=color))
    stage.scene.add(w)
    stage.scene.play(t.animate.set_value(t_end), run_time=t_end,
                     rate_func=lambda x: x)
    stage._objects.append(w)
    return w, t


def superpose_waves(stage: Stage, ax, parts, t_end: float = 4.0):
    """Two or more waves moving together, and their sum drawn in white.

    ``parts`` is a list of (amp, k, omega)."""
    _need(ax, "plot", "the axes from draw_axes(stage)", "superpose_waves(stage, axes, parts)")
    t = ValueTracker(0)
    xr = [ax.x_range[0], ax.x_range[1]]
    one = lambda a, k, w: (lambda x: a * np.sin(k * x - w * t.get_value()))
    curves = [always_redraw(lambda a=a, k=k, w=w, c=c: ax.plot(
        one(a, k, w), x_range=xr, color=c, stroke_opacity=0.6))
        for (a, k, w), c in zip(parts, _PALETTE)]
    total = always_redraw(lambda: ax.plot(
        lambda x: sum(one(a, k, w)(x) for a, k, w in parts), x_range=xr,
        color=WHITE, stroke_width=4))
    stage.scene.add(*curves, total)
    stage.scene.play(t.animate.set_value(t_end), run_time=t_end,
                     rate_func=lambda x: x)
    stage._objects += [*curves, total]
    return curves, total


def build_fourier_series(stage: Stage, ax, n_terms: int = 7, run_time: float = 1.0):
    """The square wave built from odd sines, one more term each step."""
    _need(ax, "plot", "the axes from draw_axes(stage)", "build_fourier_series(stage, axes)")
    xr = [ax.x_range[0], ax.x_range[1]]
    target = ax.plot(lambda x: np.sign(np.sin(x)), x_range=xr, color=GREY,
                     stroke_opacity=0.5, use_smoothing=False)
    stage.scene.play(Create(target), run_time=1.0)

    def partial(n):
        return lambda x: 4 / PI * sum(np.sin((2 * j + 1) * x) / (2 * j + 1)
                                      for j in range(n))

    cur = ax.plot(partial(1), x_range=xr, color=YELLOW)
    stage.scene.play(Create(cur), run_time=run_time)
    for n in range(2, n_terms + 1):
        stage.scene.play(Transform(cur, ax.plot(partial(n), x_range=xr,
                                                color=YELLOW)), run_time=run_time)
    stage._objects += [target, cur]
    return cur


def fill_halving_squares(stage: Stage, n: int = 7, where: str = "center"):
    """A unit square filled by halves: 1/2, 1/4, 1/8, ... -- the sum is 1."""
    cx, cy, w, h = _REGIONS[where]
    side = min(w, h) * 0.85
    outline = Square(side, color=WHITE).move_to([cx, cy, 0])
    stage.scene.play(Create(outline), run_time=0.8)
    x0, y0 = cx - side / 2, cy - side / 2
    wd, ht, pieces = side, side, []
    for k in range(n):
        if k % 2 == 0:
            r = Rectangle(width=wd / 2, height=ht, fill_opacity=0.7,
                          color=_PALETTE[k % len(_PALETTE)], stroke_width=1)
            r.move_to([x0 + wd / 4, y0 + ht / 2, 0])
            x0 += wd / 2
            wd /= 2
        else:
            r = Rectangle(width=wd, height=ht / 2, fill_opacity=0.7,
                          color=_PALETTE[k % len(_PALETTE)], stroke_width=1)
            r.move_to([x0 + wd / 2, y0 + ht / 4, 0])
            y0 += ht / 2
            ht /= 2
        tag = MathTex(r"\tfrac{1}{%d}" % (2 ** (k + 1)),
                      font_size=max(40 - 5 * k, 14)).move_to(r)
        stage.scene.play(FadeIn(r), FadeIn(tag), run_time=0.5)
        pieces.append(VGroup(r, tag))
    grp = VGroup(outline, *pieces)
    stage._objects.append(grp)
    return grp


def narrow_epsilon_band(stage: Stage, ax, g, limit: float, eps: float = 0.5,
                 shrink_to: float = 0.1, color=GREEN):
    """A band limit +- eps around the curve's limit, narrowing: the curve
    eventually stays inside every band."""
    _need(ax, "c2p", "the axes from draw_axes(stage)", "narrow_epsilon_band(stage, axes, graph, L)")
    e = ValueTracker(eps)
    x0, x1 = ax.x_range[0], ax.x_range[1]
    band = always_redraw(lambda: Polygon(
        ax.c2p(x0, limit - e.get_value()), ax.c2p(x1, limit - e.get_value()),
        ax.c2p(x1, limit + e.get_value()), ax.c2p(x0, limit + e.get_value()),
        color=color, fill_opacity=0.2, stroke_width=1))
    stage.scene.play(FadeIn(band), run_time=0.8)
    stage.scene.play(e.animate.set_value(shrink_to), run_time=2.5)
    stage._objects.append(band)
    return band


# -- algorithms ----------------------------------------------------------------

def draw_array(stage: Stage, values, where: str = "center", color=BLUE):
    """An array as bars, left to right, heights by value."""
    return draw_bars(stage, values, labels=values, where=where, color=color)


def swap_bars(stage: Stage, bars_, i: int, j: int, run_time: float = 0.6):
    """Swap two bars of an array_bars group, sliding past each other."""
    items = bars_[0] if isinstance(bars_[0], VGroup) and len(bars_) == 2 else bars_
    a, b = items[i], items[j]
    xa, xb = a.get_x(), b.get_x()
    stage.scene.play(a.animate.set_x(xb), b.animate.set_x(xa),
                     run_time=run_time)
    items.submobjects[i], items.submobjects[j] = b, a
    return bars_


# -- chance --------------------------------------------------------------------

def flip_coins(stage: Stage, n: int = 30, p: float = 0.5, seed: int = 1,
               where: str = "center"):
    """Flips appear as heads (yellow) and tails (blue); the share of heads is
    read out as it settles towards p."""
    rng = np.random.default_rng(seed)
    cx, cy, w, h = _REGIONS[where]
    per_row = 10
    coins, heads = VGroup(), 0
    share = DecimalNumber(0, num_decimal_places=2, font_size=36)
    read = VGroup(MathTex(r"\text{heads} =", font_size=36), share).arrange(RIGHT)
    read.move_to([cx, cy + h / 2 - 0.3, 0])
    stage.scene.play(FadeIn(read), run_time=0.4)
    for k in range(n):
        hit = rng.random() < p
        heads += hit
        c = Circle(radius=0.2, fill_opacity=0.9, stroke_width=1,
                   color=YELLOW if hit else BLUE)
        c.move_to([cx - (per_row - 1) * 0.25 + (k % per_row) * 0.5,
                   cy + h / 2 - 1.1 - (k // per_row) * 0.5, 0])
        coins.add(c)
        stage.scene.play(FadeIn(c, scale=0.5), share.animate.set_value(heads / (k + 1)),
                         run_time=0.12 if k > 5 else 0.3)
    stage._objects += [coins, read]
    return coins, read


# -- the second batch: what the hard titles ask for ----------------------------

def bayes_square(stage: Stage, prior: float = 0.01, sensitivity: float = 0.9,
                 false_pos: float = 0.09, where: str = "center"):
    """Bayes as areas: a population square split by the prior, each part split
    by the test, and the positives picked out -- the posterior is the share
    of the highlighted area that is the condition."""
    cx, cy, w, h = _REGIONS[where]
    side = min(w, h) * 0.85
    x0, y0 = cx - side / 2, cy - side / 2
    sick_w = max(side * prior, 0.08)
    well_w = side - sick_w

    def block(x, width, frac, color):
        top = Rectangle(width=width, height=side * frac, fill_opacity=0.85,
                        color=color, stroke_width=1)
        top.move_to([x + width / 2, y0 + side - side * frac / 2, 0])
        rest = Rectangle(width=width, height=side * (1 - frac), fill_opacity=0.2,
                         color=color, stroke_width=1)
        rest.move_to([x + width / 2, y0 + side * (1 - frac) / 2, 0])
        return top, rest

    outline = Square(side, color=WHITE).move_to([cx, cy, 0])
    stage.scene.play(Create(outline), run_time=0.6)
    s_pos, s_neg = block(x0, sick_w, sensitivity, RED)
    w_pos, w_neg = block(x0 + sick_w, well_w, false_pos, BLUE)
    stage.scene.play(FadeIn(s_pos), FadeIn(s_neg), FadeIn(w_pos), FadeIn(w_neg),
                     run_time=1.2)
    post = prior * sensitivity / (prior * sensitivity + (1 - prior) * false_pos)
    tag = MathTex(r"P(\text{sick}\mid +) = %.2f" % post, font_size=36)
    tag.next_to(outline, RIGHT, buff=0.3)
    stage.scene.play(Circumscribe(VGroup(s_pos, w_pos), color=YELLOW),
                     FadeIn(tag), run_time=1.2)
    grp = VGroup(outline, s_pos, s_neg, w_pos, w_neg, tag)
    stage._objects.append(grp)
    return grp


def gradient_descent(stage: Stage, ax, f, x0: float, lr: float = 0.2,
                     steps: int = 10, color=YELLOW):
    """A ball stepping downhill on f by the slope, step by step."""
    _need(ax, "c2p", "the axes from draw_axes(stage)", "gradient_descent(stage, ax, f, x0)")
    h = 1e-4
    x = x0
    ball = Dot(ax.c2p(x, f(x)), color=color, radius=0.12)
    stage.scene.play(GrowFromCenter(ball), run_time=0.5)
    for _ in range(steps):
        slope = (f(x + h) - f(x - h)) / (2 * h)
        x = x - lr * slope
        stage.scene.play(ball.animate.move_to(ax.c2p(x, f(x))), run_time=0.4)
    stage._objects.append(ball)
    return ball


def convolve_bars(stage: Stage, a, b, where: str = "center"):
    """Slide b across a; at each shift the products sum to one output bar."""
    a, b = list(a), list(b)
    out = np.convolve(a, b)
    top = max(max(a), max(b), 1e-9)
    cx, cy, w, h = _REGIONS[where]
    unit = min(w / (len(a) + 2 * len(b)), 0.7)
    base_a = cy + 0.6
    bars_a = VGroup(*[Rectangle(width=unit * 0.8, height=max(v / top * 1.4, 0.02),
                                fill_opacity=0.8, color=BLUE, stroke_width=0)
                      .move_to([cx + (i - len(a) / 2) * unit, base_a, 0],
                               aligned_edge=DOWN) for i, v in enumerate(a)])
    stage.scene.play(FadeIn(bars_a), run_time=0.6)
    otop = max(out.max(), 1e-9)
    base_o = cy - h / 2 + 0.2
    outs = VGroup()
    for k in range(len(out)):
        bars_b = VGroup(*[Rectangle(width=unit * 0.8, height=max(v / top * 1.0, 0.02),
                                    fill_opacity=0.6, color=YELLOW, stroke_width=0)
                          .move_to([cx + (k - j - len(a) / 2) * unit, base_a - 1.2, 0],
                                   aligned_edge=DOWN) for j, v in enumerate(b)])
        o = Rectangle(width=unit * 0.8, height=max(out[k] / otop * 1.6, 0.02),
                      fill_opacity=0.9, color=GREEN, stroke_width=0)
        o.move_to([cx + (k - len(a) / 2) * unit, base_o, 0], aligned_edge=DOWN)
        stage.scene.play(FadeIn(bars_b), run_time=0.2)
        stage.scene.play(GrowFromCenter(o), FadeOut(bars_b), run_time=0.3)
        outs.add(o)
    grp = VGroup(bars_a, outs)
    stage._objects.append(grp)
    return grp


def project_vector(stage: Stage, plane_, v, w, color=GREEN):
    """The shadow of v on w's line: the dot product is its length times |w|."""
    _need(plane_, "c2p", "the plane from draw_plane(stage)", "project_vector(stage, p, v, w)")
    v, w = np.array(v, dtype=float), np.array(w, dtype=float)
    av = draw_vector(stage, plane_, tuple(v), YELLOW)
    aw = draw_vector(stage, plane_, tuple(w), BLUE)
    draw_span(stage, plane_, tuple(w), color=BLUE)
    proj = (v @ w) / (w @ w) * w
    drop = DashedLine(plane_.c2p(*v), plane_.c2p(*proj), color=GREY_B)
    shadow = Arrow(plane_.c2p(0, 0), plane_.c2p(*proj), buff=0, color=color)
    stage.scene.play(Create(drop), GrowArrow(shadow), run_time=1.0)
    stage._objects += [drop, shadow]
    return av, aw, shadow


def basis_grid(stage: Stage, plane_, b1, b2, color=TEAL):
    """Another coordinate system over the same plane: the grid spanned by
    b1 and b2, drawn over the standard one."""
    _need(plane_, "c2p", "the plane from draw_plane(stage)", "basis_grid(stage, p, b1, b2)")
    b1, b2 = np.array(b1, dtype=float), np.array(b2, dtype=float)
    lines = VGroup()
    for k in range(-4, 5):
        for d, other in ((b1, b2), (b2, b1)):
            p0 = k * other - 6 * d
            p1 = k * other + 6 * d
            lines.add(Line(plane_.c2p(*p0), plane_.c2p(*p1), color=color,
                           stroke_width=1.5, stroke_opacity=0.7))
    stage.scene.play(Create(lines), run_time=1.5)
    a1 = draw_vector(stage, plane_, tuple(b1), GREEN)
    a2 = draw_vector(stage, plane_, tuple(b2), RED)
    stage._objects.append(lines)
    return lines, a1, a2


def wind_signal(stage: Stage, freqs=(3,), wind_from: float = 0.5,
                wind_to: float = 3.5, run_time: float = 5.0):
    """The Fourier machine: a signal wound around a circle at a changing
    winding frequency; its centre of mass jumps out when the winding matches
    a frequency in the signal."""
    signal = lambda t: 1 + sum(np.cos(TAU * f * t) for f in freqs) / len(freqs)
    centre = np.array([0.0, -0.3, 0])
    scale = 1.3
    wf = ValueTracker(wind_from)

    def curve():
        ts = np.linspace(0, 4.5, 600)
        pts = [centre + scale * signal(t) * np.array(
            [np.cos(-TAU * wf.get_value() * t), np.sin(-TAU * wf.get_value() * t), 0])
            for t in ts]
        from manim import VMobject
        c = VMobject(color=YELLOW, stroke_width=2)
        c.set_points_smoothly(pts)
        return c

    def mass():
        ts = np.linspace(0, 4.5, 600)
        z = np.mean([signal(t) * np.exp(-1j * TAU * wf.get_value() * t) for t in ts])
        return Dot(centre + scale * np.array([z.real, z.imag, 0]), color=RED,
                   radius=0.1)

    wound = always_redraw(curve)
    dot = always_redraw(mass)
    readout = DecimalNumber(wind_from, num_decimal_places=2, font_size=34)
    readout.add_updater(lambda m: m.set_value(wf.get_value()))
    lab = VGroup(MathTex(r"\text{winding} =", font_size=34), readout).arrange(RIGHT)
    lab.move_to([4.2, 2.3, 0])
    stage.scene.add(wound, dot)
    stage.scene.play(FadeIn(lab), run_time=0.4)
    stage.scene.play(wf.animate.set_value(wind_to), run_time=run_time,
                     rate_func=lambda x: x)
    stage._objects += [wound, dot, lab]
    return wound, dot


def prime_spiral(stage: Stage, n: int = 2000, where: str = "center"):
    """Primes plotted at (p, p) in polar coordinates: arms appear."""
    cx, cy, w, h = _REGIONS[where]
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for k in range(2, int(n ** 0.5) + 1):
        if sieve[k]:
            sieve[k * k::k] = False
    primes = np.nonzero(sieve)[0]
    r = min(w, h) / 2 / np.sqrt(n)
    dots = VGroup(*[Dot([cx + r * np.sqrt(q) * np.cos(q), cy + r * np.sqrt(q) * np.sin(q), 0],
                        radius=0.025, color=TEAL) for q in primes])
    stage.scene.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.002),
                     run_time=3.0)
    stage._objects.append(dots)
    return dots


def diffuse_heat(stage: Stage, ax, f0, t_end: float = 1.5, run_time: float = 4.0,
                 color=ORANGE):
    """A temperature curve smoothing out under the heat equation (the high
    frequencies decay fastest)."""
    _need(ax, "plot", "the axes from draw_axes(stage)", "diffuse_heat(stage, ax, f0)")
    x0, x1 = ax.x_range[0], ax.x_range[1]
    L = x1 - x0
    xs = np.linspace(x0, x1, 400)
    ys = np.array([f0(x) for x in xs])
    ks = np.arange(1, 30)
    coef = [2 / len(xs) * np.sum(ys * np.sin(k * PI * (xs - x0) / L)) for k in ks]
    t = ValueTracker(0)
    u = lambda x: sum(c * np.exp(-(k * PI / L) ** 2 * t.get_value())
                      * np.sin(k * PI * (x - x0) / L) for c, k in zip(coef, ks))
    curve = always_redraw(lambda: ax.plot(u, x_range=[x0, x1], color=color))
    stage.scene.play(Create(curve), run_time=1.0)
    stage.scene.play(t.animate.set_value(t_end), run_time=run_time)
    stage._objects.append(curve)
    return curve


def hanoi_moves(stage: Stage, n: int = 3, where: str = "center"):
    """Towers of Hanoi, solved move by move."""
    cx, cy, w, h = _REGIONS[where]
    xs = {"A": cx - w / 3, "B": cx, "C": cx + w / 3}
    base = cy - h / 2 + 0.5
    pegs = VGroup(*[Line([x, base, 0], [x, base + 2.4, 0], color=GREY_B,
                         stroke_width=6) for x in xs.values()])
    stage.scene.play(Create(pegs), run_time=0.6)
    discs = {k: Rectangle(width=0.6 + 0.35 * k, height=0.28, fill_opacity=0.9,
                          color=_PALETTE[k % len(_PALETTE)], stroke_width=1)
             for k in range(n, 0, -1)}
    stacks = {"A": list(range(n, 0, -1)), "B": [], "C": []}
    for i, k in enumerate(stacks["A"]):
        discs[k].move_to([xs["A"], base + 0.14 + 0.3 * i, 0])
    stage.scene.play(*[FadeIn(d) for d in discs.values()], run_time=0.6)

    def solve(m, a, c, b):
        if m == 0:
            return
        solve(m - 1, a, b, c)
        k = stacks[a].pop()
        stacks[c].append(k)
        d = discs[k]
        stage.scene.play(d.animate.move_to([xs[a], base + 2.8, 0]), run_time=0.2)
        stage.scene.play(d.animate.move_to([xs[c], base + 2.8, 0]), run_time=0.2)
        stage.scene.play(d.animate.move_to([xs[c], base + 0.14 + 0.3 * (len(stacks[c]) - 1), 0]),
                         run_time=0.2)
        solve(m - 1, b, c, a)

    solve(n, "A", "C", "B")
    grp = VGroup(pegs, *discs.values())
    stage._objects.append(grp)
    return grp


def bit_grid(stage: Stage, bits, where: str = "center", highlight_cols=None,
             highlight_rows=None):
    """A 4x4 grid of bits (Hamming codes); optional parity rows/columns lit."""
    bits = list(bits)[:16] + [0] * max(0, 16 - len(bits))
    cells = VGroup()
    for k, b in enumerate(bits):
        sq = Square(0.8, stroke_width=1.5, color=GREY_B)
        sq.add(MathTex(str(b), font_size=36))
        if b:
            sq.set_fill(BLUE, opacity=0.35)
        cells.add(sq)
    cells.arrange_in_grid(4, 4, buff=0)
    stage.place(cells, where)
    stage.scene.play(LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.03),
                     run_time=1.0)
    lit = []
    for c in highlight_cols or []:
        lit += [cells[r * 4 + c] for r in range(4)]
    for r in highlight_rows or []:
        lit += [cells[r * 4 + c] for c in range(4)]
    if lit:
        stage.scene.play(*[m.animate.set_stroke(YELLOW, width=4) for m in lit],
                         run_time=0.8)
    return cells


def flow_particles(stage: Stage, f, n: int = 60, run_time: float = 4.0, seed: int = 0):
    """Particles carried along a 2D field f(x, y) -> (u, v): sources spread,
    sinks gather, curl swirls."""
    rng = np.random.default_rng(seed)
    starts = np.column_stack([rng.uniform(-5, 5, n), rng.uniform(-3, 3, n)])
    dots = VGroup(*[Dot([x, y, 0], radius=0.05, color=YELLOW) for x, y in starts])

    def carry(mob, dt):
        for d in mob:
            x, y, _ = d.get_center()
            u, v = f(x, y)
            d.shift(dt * 0.6 * np.array([u, v, 0]))

    stage.scene.add(dots)
    dots.add_updater(carry)
    stage.scene.wait(run_time)
    dots.clear_updaters()
    stage._objects.append(dots)
    return dots


def euler_circle(stage: Stage, cp, t_end: float = TAU, run_time: float = 4.0):
    """e^{it} walking the unit circle as t grows, t read out."""
    _need(cp, "n2p", "the plane from draw_complex_plane(stage)", "euler_circle(stage, cp)")
    t = ValueTracker(0)
    circle = Circle(radius=abs(cp.n2p(1)[0] - cp.n2p(0)[0]), color=GREY_B)
    circle.move_to(cp.n2p(0))
    arrow = always_redraw(lambda: Arrow(cp.n2p(0), cp.n2p(np.exp(1j * t.get_value())),
                                        buff=0, color=YELLOW))
    read = DecimalNumber(0, num_decimal_places=2, font_size=34)
    read.add_updater(lambda m: m.set_value(t.get_value()))
    lab = VGroup(MathTex(r"e^{it},\ t =", font_size=34), read).arrange(RIGHT)
    lab.move_to(cp.n2p(0) + np.array([3.2, 2.2, 0]))
    stage.scene.play(Create(circle), FadeIn(lab), run_time=0.8)
    stage.scene.add(arrow)
    stage.scene.play(t.animate.set_value(t_end), run_time=run_time,
                     rate_func=lambda x: x)
    stage._objects += [circle, arrow, lab]
    return arrow


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
Blocks are verbs (draw_plane, plot_graph); keep what they return in short
nouns (p, ax, g, v) and animate those -- never a block's name.
  stage.title(s)  stage.caption(s)   -- replace the title / caption slot
  stage.equation(tex1, tex2, ..., where="right") -- a derivation, each step
      transforming into the next
  stage.label(m, s)  stage.clear(keep=[...])  stage.pause(seconds)
  draw_plane(stage, where="center", x_extent=4, y_extent=3) -> NumberPlane
  draw_vector(stage, p, (x, y), color=YELLOW, label=None) -> Arrow
  draw_basis(stage, p) -> (i_hat, j_hat)
  apply_matrix(stage, p, [[a, b], [c, d]], riders=[arrows...])
  draw_unit_square(stage, p) -> Polygon      draw_span(stage, p, (x, y)) -> Line
  scale_vector(stage, p, arrow, factor)
  draw_axes(stage, x_range=(a, b), y_range=(c, d), where="center") -> Axes
  plot_graph(stage, ax, f, color=BLUE, label=None) -> graph   (f is a lambda)
  slide_tangent(stage, ax, f, x_start, x_end)   -- slides, slope read out live
  shade_area(stage, ax, g, a, b)   riemann_refine(stage, ax, g, a, b, ns=(4, 8, 16, 32))
  trace_graph(stage, ax, f, x_start, x_end)   -- a dot runs along f
  draw_number_line(stage, x_range=(a, b)) -> NumberLine
  mark_point(stage, nl, x, label=None) -> Dot
  draw_bars(stage, values, labels=None) -> bars
  show_partial_sums(stage, term, n=10) -> (bars, readout)  (term is a lambda k: ...)
  slice_circle(stage, n=12) -> sectors   unroll_slices(stage, sectors)
  draw_right_triangle(stage, a=3, b=2) -> triangle
  draw_dice_grid(stage, highlight_sum=None) -> cells
  show_determinant(stage, p, [[a, b], [c, d]]) -> (square, area_label)
  show_eigenvectors(stage, p, [[a, b], [c, d]])  -- eigen-directions stay put
  taylor_approximate(stage, ax, f, a, [f(a), f'(a), f''(a), ...])  -- polynomials close in
  circle_to_sine(stage)   -- a turning radius draws a sine wave
  draw_vector_field(stage, lambda x, y: (u, v))
  draw_complex_plane(stage) -> cp    multiply_complex(stage, cp, z)  -- rotate and scale
  draw_neural_net(stage, layers=(3, 4, 2)) -> net
  draw_network(stage, {"A": (x, y), ...}, [("A", "B"), ...]) -> (dots, lines)
  grow_histogram(stage, lambda rng, k: rng.normal(size=k), bins=[...])
  animate_wave(stage, ax, amp=1, k=2, omega=2)   superpose_waves(stage, ax, [(a, k, w), ...])
  build_fourier_series(stage, ax, n_terms=7)   -- a square wave from odd sines
  fill_halving_squares(stage, n=7)            -- 1/2 + 1/4 + ... fills the square
  narrow_epsilon_band(stage, ax, g, L, eps=0.5) -- the band narrows around the limit
  draw_array(stage, [5, 2, 8, ...]) -> bars   swap_bars(stage, bars, i, j)
  flip_coins(stage, n=30, p=0.5)         -- share of heads settles
  bayes_square(stage, prior=0.01, sensitivity=0.9, false_pos=0.09)
  gradient_descent(stage, ax, f, x0, lr=0.2, steps=10) -- a ball steps downhill
  convolve_bars(stage, [a...], [b...])   -- b slides over a, sums build output
  project_vector(stage, p, v, w)         -- v's shadow on w's line
  basis_grid(stage, p, b1, b2)           -- another basis's grid over the plane
  wind_signal(stage, freqs=(3,))         -- Fourier: wind a signal round a circle
  prime_spiral(stage, n=2000)            -- primes at (p, p) in polar
  diffuse_heat(stage, ax, lambda x: ...) -- a temperature curve smooths out
  hanoi_moves(stage, n=3)   bit_grid(stage, bits, highlight_cols=[1, 3])
  flow_particles(stage, lambda x, y: (u, v))  -- particles ride a field
  euler_circle(stage, cp)                -- e^{it} walks the unit circle
  highlight(stage, m)   pulse(stage, m)
Regions: "center", "left", "right", "full". Text only through stage.title,
stage.caption, stage.label and stage.equation.
"""


#: Every block that draws something (everything taking a stage, bar the two
#: emphasis helpers), and those that also move it. Scorecards and the GRPO
#: reward read these instead of keeping their own lists, which went stale
#: the day the kit grew.
KIT_BLOCKS = {n for n, f in list(globals().items())
              if callable(f) and not n.startswith("_") and not isinstance(f, type)
              and getattr(f, "__code__", None) is not None
              and f.__code__.co_varnames[:1] == ("stage",)
              and n not in {"highlight", "pulse"}}
KIT_MOVES = {"apply_matrix", "slide_tangent", "riemann_refine", "trace_graph",
             "scale_vector", "unroll_slices", "show_eigenvectors",
             "show_determinant", "taylor_approximate", "circle_to_sine",
             "multiply_complex", "grow_histogram", "animate_wave",
             "superpose_waves", "build_fourier_series", "narrow_epsilon_band",
             "swap_bars", "flip_coins", "gradient_descent", "convolve_bars",
             "wind_signal", "diffuse_heat", "hanoi_moves", "flow_particles",
             "euler_circle"}
