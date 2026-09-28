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


def _pad_init(cls, point_kw=()):
    """Let a constructor take 2D points: Polygon((0, 0), (1, 0), (0, 1)),
    Line((0, 0), (2, 1)), Dot((1, 2)) failed with "could not broadcast
    input array from shape (1,2) into shape (1,3)"."""
    original = cls.__init__

    def init(self, *args, **kw):
        args = tuple(_as_point(a) if np.ndim(a) == 1 else a for a in args)
        for k in point_kw:
            if k in kw and np.ndim(kw[k]) == 1:
                kw[k] = _as_point(kw[k])
        original(self, *args, **kw)
    cls.__init__ = init


if not getattr(_Mobject, "_forge_2d_ok", False):
    for _c in (Line, Arrow, DashedLine, Dot, Polygon):
        _pad_init(_c, ("start", "end", "point"))
    _pad_points("shift")
    _pad_points("move_to", first_only=True)
    _pad_points("next_to", first_only=True)
    _Mobject._forge_2d_ok = True


# -- the stage -----------------------------------------------------------------

class _Regions(dict):
    """The four regions, forgiving about how one is named: "LEFT",
    "top left", a direction constant or an (x, y) point go to the nearest
    region instead of raising KeyError deep inside a block."""

    def __getitem__(self, where):
        if isinstance(where, str):
            w = where.strip().lower()
            if dict.__contains__(self, w):
                return dict.__getitem__(self, w)
            key = "left" if "left" in w else "right" if "right" in w else \
                "full" if w in ("all", "whole", "screen", "full screen") else "center"
            return dict.__getitem__(self, key)
        try:
            x = float(np.asarray(where, dtype=float).ravel()[0])
        except (TypeError, ValueError, IndexError):
            return dict.__getitem__(self, "center")
        key = "left" if x < -0.5 else "right" if x > 0.5 else "center"
        return dict.__getitem__(self, key)


_REGIONS = _Regions({
    # centre x, centre y, width, height, in frame units (14.2 x 8)
    "center": (0.0, -0.1, 9.0, 5.4),
    "left": (-3.4, -0.1, 6.2, 5.4),
    "right": (3.4, -0.1, 6.2, 5.4),
    "full": (0.0, -0.1, 13.0, 5.8),
})
_PALETTE = [BLUE, YELLOW, GREEN, RED, TEAL, ORANGE]
_CURRENT: list = []      # the live Stage, for block calls that forget it


class Stage:
    """Owns the layout: a title slot, a caption slot, and regions between.

    ``stage.title("...")`` and ``stage.caption("...")`` replace what is in
    their slot; ``stage.place(m, "left")`` scales and moves a mobject into a
    region; ``stage.clear()`` fades out everything but the title.
    """

    def __init__(self, scene):
        _CURRENT[:] = [self]
        self.scene = scene
        self._last: dict = {}       # newest plane / axes / graph, by slot
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
                if stop in s and len(s.split(stop)[0].split()) >= 3:
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

    def _make_room(self, where: str, run_time: float = 0.8):
        """Move the picture out of a side region about to be written in.

        Models draw axes or a plane in the centre and then write an
        equation "right", on top of the graph (the derivative scene's limit
        formula sat across the curve). The stage's pictures that overlap the
        region are scaled into the opposite side first.
        """
        if where not in ("left", "right"):
            return
        from manim import Group
        cx, _, w, _ = _REGIONS[where]
        # Everything on screen but the title and caption: blocks that fade
        # their parts in one by one never put their returned group on the
        # scene, so moving only the stage's own objects left bars, networks
        # and dots behind while their labels moved (the teacher batches).
        # Not the timers: a ValueTracker animated with .animate sits on the
        # scene at x = its value (400) and scaled the picture to nothing.
        tops = [m for m in self.scene.mobjects
                if m is not self._title and m is not self._caption
                and not isinstance(m, ValueTracker)
                and (m.width > 1e-6 or m.height > 1e-6 or m.submobjects)]
        if not tops:
            return
        # Sized by what is on the frame: a sheared plane's grid runs far
        # past it and made the picture shrink to nothing.
        sized = [m for m in tops if m.width < 16 and m.height < 10] or tops
        box = Group(*sized)
        lo, hi = box.get_left()[0], box.get_right()[0]
        if hi <= cx - w / 2 + 0.2 or lo >= cx + w / 2 - 0.2:
            return
        ox, oy, ow, oh = _REGIONS["left" if where == "right" else "right"]
        k = min(1.0, ow / max(box.width, 1e-6), oh / max(box.height, 1e-6))
        centre = box.get_center()
        # Live readouts and redrawn shapes (a slope value, a pendulum on a
        # fixed pivot) are frozen where they are: moving them while they
        # update crashed ("zip() argument 2 is longer") or pulled them back.
        for m in tops:
            for x in m.get_family():
                x.clear_updaters()
        grp = Group(*tops)
        self.scene.play(grp.animate.scale(k, about_point=centre)
                        .shift(np.array([ox, oy, 0]) - centre), run_time=run_time)
        # Unwrap: left as one group on the scene, a later clear(keep=[ax])
        # faded the group -- axes and all.
        self.scene.remove(grp)
        self.scene.add(*tops)

    def equation(self, *tex: str, where: str = "right", run_time: float = 1.2):
        """A derivation: each step transforms into the next, in a region."""
        old = getattr(self, "_equation", None)
        if old is not None and old in self.scene.mobjects:   # one at a time
            self.scene.play(FadeOut(old), run_time=0.5)
        self._make_room(where)
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
        self._equation = cur
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
        if isinstance(m, (tuple, list)) and m and not np.isscalar(m[0]):
            m = m[0]                     # a block's (picture, readout) pair
        if not hasattr(m, "get_center"):                 # a point: label there
            m = Dot(_xy(m), radius=0.001, fill_opacity=0)
        t = self._text(str(s), 28).set_color(color).next_to(m, direction, buff=0.15)
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

    def mark(self):
        """Record the scene clock at the end of a beat (the pipeline adds
        one after each beat when asked): contact sheets then show each
        beat's last frame instead of evenly spaced ones, which caught
        curves mid-draw and missed pictures that were on screen briefly."""
        import sys as _sys
        self._t = getattr(self, "_t", []) + [float(self.scene.renderer.time)]
        print("BEAT_ENDS", self._t, file=_sys.stderr, flush=True)

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
        if callable(f) and getattr(f, "_kit_block", False):
            return lambda *a, **kw: f(self, *a, **kw)
        # stage.coords_to_point(...), stage.c2p(...): the newest axes' own
        for k in ("ax", "plane_", "cp", "nl"):
            have = self.__dict__.get("_last", {}).get(k)
            if have is not None and hasattr(have, name):
                return getattr(have, name)
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


def _xy(v):
    """A 2D or 3D point (tuple, list, array, Manim constant) as a 3D point."""
    a = np.asarray(v, dtype=float).ravel()
    return np.array([a[0], a[1] if len(a) > 1 else 0.0, 0.0])


def _span(r, ticks: int = 8):
    """[lo, hi, step] from (lo, hi) or (lo, hi, step); without a step, a
    1-2-5 step giving about ``ticks`` ticks (a (0, 1000) axis had 1,000
    numbered ticks)."""
    r = [float(x) for x in r]
    lo, hi = min(r[0], r[1]), max(r[0], r[1])
    if hi == lo:
        hi = lo + 1
    if len(r) > 2 and r[2] > 0:
        return [lo, hi, r[2]]
    raw = (hi - lo) / ticks
    mag = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1
    step = next((m * mag for m in (1, 2, 5, 10) if m * mag >= raw), 10 * mag)
    return [lo, hi, max(step, 1.0) if hi - lo >= 4 else step]


def _graph(ax, g, a=None, b=None):
    """A plotted graph from either a graph or a plain function (models pass
    the lambda far more often than the graph plot_graph returned)."""
    if hasattr(g, "point_from_proportion"):
        return g
    lo, hi = ax.x_range[0], ax.x_range[1]
    if a is not None and b is not None:
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    return ax.plot(g, x_range=[lo, hi])


def _inside(ax, f, lo, hi):
    """The widest stretch of [lo, hi] where f stays inside the axes' y-range
    (with a little slack): a parabola plotted over the whole x-range ran
    off the top of the axes and through the title."""
    ylo, yhi = ax.y_range[0], ax.y_range[1]
    slack = 0.05 * (yhi - ylo)
    xs = np.linspace(lo, hi, 401)
    with np.errstate(all="ignore"):
        try:
            ys = np.array([float(f(x)) for x in xs])
        except Exception:
            return lo, hi
    ok = np.isfinite(ys) & (ys >= ylo - slack) & (ys <= yhi + slack)
    if ok.all() or not ok.any():
        return lo, hi
    best, run, start = (0, 0), 0, 0
    for i, v in enumerate(ok):
        if v:
            if run == 0:
                start = i
            run += 1
            if run > best[1] - best[0]:
                best = (start, i + 1)
        else:
            run = 0
    return xs[best[0]], xs[best[1] - 1]


_CALLABLE: dict = {}


def _callable_graph(cls):
    if cls not in _CALLABLE:
        _CALLABLE[cls] = type(cls.__name__, (cls,),
                              {"__call__": lambda self, x: self.underlying_function(x)})
    return _CALLABLE[cls]


def _readout_corner(ax, read):
    """A live readout inside the axes' top-right corner. Above the axes it
    collided with the title whenever the axes were tall."""
    read.move_to(ax.get_corner(UP + RIGHT) + np.array([-read.width / 2 - 0.1,
                                                       -read.height / 2 - 0.05, 0]))
    return read


def _field(f):
    """f(x, y) -> (u, v), from a field written either as f(x, y) or f(point)."""
    def uv(x, y):
        try:
            out = f(x, y)
        except TypeError:
            out = f(np.array([x, y, 0.0]))
        out = np.asarray(out, dtype=float).ravel()
        return out[0], (out[1] if len(out) > 1 else 0.0)
    return uv


# -- basic shapes: the names a model guesses first -----------------------------
# The first two requests in the app reached for draw_square and draw_dots,
# which did not exist; the kit had eigenvectors but no square.

def draw_square(stage: Stage, side: float = 2.0, where: str = "center",
                color=BLUE, label: str | None = None, fill: float = 0.3):
    """A square of the given side (in frame units), drawn, optionally labelled."""
    sq = Square(side_length=side, color=color, fill_opacity=fill)
    stage.place(sq, where)
    stage.scene.play(Create(sq), run_time=0.8)
    if label:
        stage.label(sq, label, direction=DOWN, color=color)
    return sq


def draw_rectangle(stage: Stage, width: float = 3.0, height: float = 2.0,
                   where: str = "center", color=BLUE, label: str | None = None,
                   fill: float = 0.3):
    """A rectangle, drawn, optionally labelled."""
    r = Rectangle(width=width, height=height, color=color, fill_opacity=fill)
    stage.place(r, where)
    stage.scene.play(Create(r), run_time=0.8)
    if label:
        stage.label(r, label, direction=DOWN, color=color)
    return r


def draw_circle(stage: Stage, radius: float = 1.5, where: str = "center",
                color=BLUE, label: str | None = None, fill: float = 0.2):
    """A circle, drawn, optionally labelled."""
    c = Circle(radius=radius, color=color, fill_opacity=fill)
    stage.place(c, where)
    stage.scene.play(Create(c), run_time=0.8)
    if label:
        stage.label(c, label, direction=DOWN, color=color)
    return c


def draw_polygon(stage: Stage, points, where: str = "center", color=BLUE,
                 fill: float = 0.3):
    """A polygon through 2D points [(x, y), ...], drawn."""
    poly = Polygon(*[_xy(q) for q in points], color=color,
                   fill_opacity=fill)
    if where is None:                  # at its own coordinates
        stage._objects.append(poly)
    else:
        stage.place(poly, where)
    stage.scene.play(Create(poly), run_time=0.8)
    return poly


def draw_dots(stage: Stage, n: int = 9, cols: int | None = None,
              where: str = "center", color=YELLOW):
    """n dots in a grid (cols per row, default near-square), appearing in turn."""
    cols = cols or max(1, int(np.ceil(np.sqrt(n))))
    dots = VGroup(*[Dot(radius=0.12, color=color) for _ in range(n)])
    dots.arrange_in_grid(cols=cols, buff=0.35)
    stage.place(dots, where)
    stage.scene.play(LaggedStart(*[GrowFromCenter(d) for d in dots],
                                 lag_ratio=0.08), run_time=1.2)
    return dots


def draw_arrow(stage: Stage, start, end, color=YELLOW, label: str | None = None):
    """An arrow between two 2D points, grown."""
    a = Arrow(_xy(start), _xy(end), buff=0, color=color)
    stage.scene.play(GrowArrow(a), run_time=0.7)
    stage._objects.append(a)
    if label:
        stage.label(a, label, color=color)
    return a


def draw_line(stage: Stage, start, end, color=WHITE, dashed: bool = False):
    """A line (or dashed line) between two 2D points, drawn."""
    cls = DashedLine if dashed else Line
    ln = cls(_xy(start), _xy(end), color=color)
    stage.scene.play(Create(ln), run_time=0.6)
    stage._objects.append(ln)
    return ln


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
        side = RIGHT if xy[0] >= 0 else LEFT
        t = stage.label(a, label, direction=side, color=color)
        # The label rides the tip: apply_matrix moves the arrow, and a label
        # left behind pointed the wrong way after a reflection.
        t.add_updater(lambda m, a=a, side=side: m.next_to(a.get_end(), side, buff=0.15))
    return a


def draw_basis(stage: Stage, plane_):
    """i-hat and j-hat, green and red, labelled."""
    _need(plane_, "c2p", "the plane from draw_plane(stage)", "draw_basis(stage, plane)")
    i = draw_vector(stage, plane_, (1, 0), GREEN, r"\hat{\imath}")
    j = draw_vector(stage, plane_, (0, 1), RED, r"\hat{\jmath}")
    return i, j


def apply_matrix(stage: Stage, plane_, matrix=((1, 1), (0, 1)), riders=(),
                 run_time: float = 2.0):
    """Move the grid, and anything riding on it, by a 2x2 matrix."""
    _need(plane_, "c2p", "the plane from draw_plane(stage)", "apply_matrix(stage, plane, matrix)")
    m = np.array(matrix, dtype=float)
    # riders given as (x, y) points: vectors drawn for them first
    riders = list(riders or [])
    if riders and not hasattr(riders[0], "get_center") and np.ndim(riders[0]) == 0:
        riders = [riders]                                # one (x, y) alone
    riders = [r if hasattr(r, "get_center") else draw_vector(stage, plane_, tuple(r))
              for r in riders]
    about = plane_.c2p(0, 0)
    group = VGroup(plane_, *riders)
    stage.scene.play(ApplyMatrix(m, group, about_point=about), run_time=run_time)
    # Unwrapped: left as one group wider than the frame, a later equation
    # sized the picture by the leftover labels and put shapes off screen.
    stage.scene.remove(group)
    stage.scene.add(plane_, *riders)
    # A stretching matrix carried arrows off the frame and the grid across
    # the title (held-out "matrix multiplication"; three teacher batches).
    # If the riders now leave the picture area, zoom the whole view out
    # about the origin until they fit -- the grid scales with them, so the
    # picture stays true, only smaller.
    if riders:
        box = VGroup(*riders)
        lim_x, lim_y = 6.4, 2.9
        far = max(np.abs(box.get_corner(UP + RIGHT) - about)[0], np.abs(box.get_corner(DOWN + LEFT) - about)[0],
                  1e-6) / lim_x
        far_y = max(np.abs(box.get_corner(UP + RIGHT) - about)[1], np.abs(box.get_corner(DOWN + LEFT) - about)[1],
                    1e-6) / lim_y
        k = 1 / max(far, far_y)
        if k < 0.95:
            stage.scene.play(group.animate.scale(k, about_point=about), run_time=0.8)
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
    d = np.array([*xy[:2], 0.0]) / (np.linalg.norm(xy[:2]) or 1)
    # Out to the plane's edge, not ±8 units through the title.
    xr = getattr(plane_, "x_range", (-4, 4))
    yr = getattr(plane_, "y_range", (-3, 3))
    reach = min(abs(xr[1]) / (abs(d[0]) or 1e-9), abs(yr[1]) / (abs(d[1]) or 1e-9))
    ln = Line(plane_.c2p(*(-reach * d[:2])), plane_.c2p(*(reach * d[:2])),
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
    xs, ys = _span(x_range), _span(y_range)

    def places(r):                    # "12", not "12.0"; "0.25" keeps two
        step = r[2]
        if all(float(v).is_integer() for v in r):
            return 0
        return 1 if float(step * 10).is_integer() else 2
    ax = Axes(x_range=xs, y_range=ys, x_length=8, y_length=5, tips=False,
              axis_config={"include_numbers": True, "font_size": 24},
              x_axis_config={"decimal_number_config": {"num_decimal_places": places(xs)}},
              y_axis_config={"decimal_number_config": {"num_decimal_places": places(ys)}})
    stage.place(ax, where)
    lab = ax.get_axis_labels(MathTex(labels[0]), MathTex(labels[1]))
    stage.scene.play(Create(ax), FadeIn(lab), run_time=1.5)
    # Part of the axes: moved with them, and kept by clear(keep=[ax]).
    stage.scene.remove(lab)
    ax.add(lab)
    return ax


def plot_graph(stage: Stage, ax, f, x_range=None, color=BLUE, label: str | None = None):
    """Plot f on the axes and draw it."""
    _need(ax, "plot", "the axes from draw_axes(stage)", "plot_graph(stage, axes, f)")
    xr = x_range or (ax.x_range[0], ax.x_range[1])
    xr = _inside(ax, f, xr[0], xr[1])
    g = ax.plot(f, x_range=[xr[0], xr[1]], color=color)
    # Callable like the function it plots: models write g(x) for a height
    # on the curve ("'ParametricFunction' object is not callable", twice).
    g.__class__ = _callable_graph(type(g))
    stage.scene.play(Create(g), run_time=1.5)
    stage._objects.append(g)
    if label:
        t = MathTex(label, color=color, font_size=32).next_to(
            g.get_end(), UP + RIGHT, buff=0.1)
        stage.scene.play(FadeIn(t), run_time=0.5)
        stage._objects.append(t)
    return g


def slide_tangent(stage: Stage, ax, f, x_start: float | None = None,
                  x_end: float | None = None, color=YELLOW, run_time: float = 3.0):
    """A tangent line sliding along f, with its slope read out live."""
    lo, hi = ax.x_range[0], ax.x_range[1]
    if x_start is None:
        x_start = lo + 0.2 * (hi - lo)
    if x_end is None:
        x_end = min(hi, x_start + 0.5 * (hi - lo))
    _need(ax, "c2p", "the axes from draw_axes(stage)", "slide_tangent(stage, axes, f, x0, x1)")
    t = ValueTracker(x_start)
    h = 1e-4

    def line():
        x = t.get_value()
        k = (f(x + h) - f(x - h)) / (2 * h)
        d = 1.2 / np.sqrt(1 + k * k)   # steep tangents ran through the title
        p0 = ax.c2p(x - d, f(x) - d * k)
        p1 = ax.c2p(x + d, f(x) + d * k)
        return Line(p0, p1, color=color)

    tan = always_redraw(line)
    dot = always_redraw(lambda: Dot(ax.c2p(t.get_value(), f(t.get_value())),
                                    color=color))
    # Starts at the real slope and updates only once shown: a readout that
    # gained a minus sign mid-FadeIn changed its glyph count, and Manim's
    # interpolation failed with "zip() argument 2 is longer than argument 1"
    # on every tangent that started on a falling stretch.
    slope = DecimalNumber((f(x_start + h) - f(x_start - h)) / (2 * h),
                          num_decimal_places=2, font_size=34)
    read = VGroup(MathTex("\\text{slope} =", font_size=34), slope).arrange(RIGHT)
    _readout_corner(ax, read)
    stage.scene.play(Create(tan), FadeIn(dot), FadeIn(read), run_time=1.0)
    slope.add_updater(lambda m: m.set_value(
        (f(t.get_value() + h) - f(t.get_value() - h)) / (2 * h)))
    stage.scene.play(t.animate.set_value(x_end), run_time=run_time)
    stage._objects += [tan, dot, read]
    return tan, dot, read


def _bounds(stage, ax, g, a, b, x_range):
    """(g, a, b) from the ways a model writes an interval: (a, b) as one
    tuple, x_range=, the bounds without the function, or nothing."""
    if b is None and x_range is None and np.ndim(a) == 1:
        a, b = a[0], a[1]
    elif b is None and x_range is not None:
        a, b = x_range[0], x_range[1]
    elif b is None and np.ndim(g) == 0 and not callable(g) and a is not None:
        g, a, b = None, g, a
    if g is None:
        g = stage._last.get("graph") or (lambda x: x)
    if a is None or b is None:
        a, b = ax.x_range[0], ax.x_range[1]
    # inside the axes: shading past the x-range drew a block off the chart
    lo, hi = ax.x_range[0], ax.x_range[1]
    a, b = (min(max(v, lo), hi) for v in (min(a, b), max(a, b)))
    if b - a < 1e-6:
        a, b = lo, hi
    return g, a, b


def shade_area(stage: Stage, ax, g=None, a=None, b=None, color=BLUE_D, x_range=None):
    """Shade the area under g between a and b."""
    _need(ax, "get_area", "the axes from draw_axes(stage)", "shade_area(stage, axes, graph, a, b)")
    g, a, b = _bounds(stage, ax, g, a, b, x_range)
    r = ax.get_area(_graph(ax, g, a, b), x_range=[a, b], color=color, opacity=0.5)
    stage.scene.play(FadeIn(r), run_time=1.0)
    stage._objects.append(r)
    return r


def riemann_refine(stage: Stage, ax, g=None, a=None, b=None, ns=(4, 8, 16, 32),
            color=TEAL, x_range=None):
    """Rectangles under g, refined: 4, 8, 16, 32 strips."""
    _need(ax, "get_riemann_rectangles", "the axes from draw_axes(stage)", "riemann_refine(stage, axes, graph, a, b)")
    g, a, b = _bounds(stage, ax, g, a, b, x_range)
    g = _graph(ax, g, a, b)
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
    xs = _span(x_range, 10)
    dp = 0 if all(float(v).is_integer() for v in xs) else \
        (1 if float(xs[2] * 10).is_integer() else 2)       # 0.25 keeps two places
    nl = NumberLine(x_range=xs, length=10, include_numbers=True,
                    decimal_number_config={"num_decimal_places": dp})
    stage.place(nl, where)
    stage.scene.play(Create(nl), run_time=1.0)
    return nl


def mark_point(stage: Stage, nl, x: float, color=YELLOW, label: str | None = None,
               direction=UP):
    """A dot at x on a number line, optionally labelled."""
    if np.ndim(x) == 1 and hasattr(nl, "c2p"):          # a point on axes
        d = Dot(nl.c2p(*list(x)[:2]), color=color)
    else:
        d = Dot(nl.n2p(float(np.ravel(x)[0]) if np.ndim(x) else x), color=color)
    stage.scene.play(GrowFromCenter(d), run_time=0.5)
    stage._objects.append(d)
    if label:
        stage.label(d, label, direction=direction, color=color)
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
        group.tags = tags
    return group


def show_partial_sums(stage: Stage, term, n: int = 10, where: str = "center",
                 color=YELLOW):
    """Bars of term(1)..term(n) stacking into a running total, read out."""
    if np.ndim(term) == 1:                    # the terms themselves, as a list
        vals = [float(v) for v in term]
        term, n = (lambda k: vals[k - 1]), len(vals)
    n = int(n) if np.ndim(n) == 0 else 10
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

def show_determinant(stage: Stage, plane_, matrix=((2, 1), (1, 2)), run_time: float = 2.0):
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


def show_eigenvectors(stage: Stage, plane_, matrix=((3, 1), (0, 2)), run_time: float = 2.5):
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
    # A test vector on neither eigen-line: (1, -1) is itself an eigenvector
    # of the y = x reflection, and then nothing visibly turned.
    lines = [np.real(vecs[:, k]) for k in range(2) if abs(np.imag(vals[k])) < 1e-9]
    off_xy = next((c for c in ((1, 1), (1, -1), (2, 1), (1, 2), (1, 0), (0, 1))
                   if all(abs(c[0] * e[1] - c[1] * e[0]) > 0.15 * np.hypot(*c) * np.hypot(*e)
                          for e in lines)), (2, 1))
    off = draw_vector(stage, plane_, off_xy, RED)
    riders.append(off)
    apply_matrix(stage, plane_, m, riders=riders, run_time=run_time)
    return riders


# -- more calculus -------------------------------------------------------------

def taylor_approximate(stage: Stage, ax, f, a: float = 0.0, terms=None, colors=None,
                       run_time: float = 1.2):
    """Taylor polynomials about a, one more term each time, closing in on f.

    ``terms`` is a list of the derivatives' values at a: [f(a), f'(a), ...].
    """
    import math
    # terms left out, or given as a count: the derivatives at a, from a
    # polynomial fitted to f around a.
    if terms is None or np.ndim(terms) == 0:
        n = 5 if terms is None else int(max(1, min(int(terms), 8)))
        xs = np.linspace(a - 1.0, a + 1.0, 81)
        fit = np.polynomial.Polynomial.fit(xs - a, [f(x) for x in xs], 12).convert()
        terms = [float(fit.deriv(k)(0.0)) if k else float(fit(0.0)) for k in range(n)]
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
    f = _field(f)
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
    net.layers, net.edges = list(cols), edges   # models reach for these
    stage._objects.append(net)
    return net


def draw_network(stage: Stage, nodes=None, edges=(), where: str = "center"):
    """A graph: nodes {name: (x, y)} in [-1, 1]^2, edges [(a, b), ...]."""
    if not isinstance(nodes, dict):          # no graph given: a small one
        nodes = {"A": (-0.8, 0.5), "B": (0, 0.8), "C": (0.8, 0.3),
                 "D": (-0.4, -0.6), "E": (0.5, -0.5)}
        edges = [("A", "B"), ("B", "C"), ("A", "D"), ("D", "E"), ("C", "E"), ("B", "E")]
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
    if np.ndim(bins) == 0:          # a bin count: edges from a pilot sample
        pilot = np.asarray(sampler(np.random.default_rng(seed + 1), 1000), dtype=float)
        edges = np.linspace(pilot.min(), pilot.max(), int(bins) + 1)
    else:
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
    """Waves parts=[(amp, k, omega), ...] moving together, their sum in white.

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
    moves = [a.animate.set_x(xb), b.animate.set_x(xa)]
    tags = getattr(bars_, "tags", None)     # draw_array's values go along
    if tags is not None and max(i, j) < len(tags):
        moves += [tags[i].animate.set_x(xb), tags[j].animate.set_x(xa)]
    stage.scene.play(*moves, run_time=run_time)
    items.submobjects[i], items.submobjects[j] = b, a
    if tags is not None and max(i, j) < len(tags):
        tags.submobjects[i], tags.submobjects[j] = tags[j], tags[i]
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
    for k in range(-3, 4):            # kept near the plane, not across the frame
        for d, other in ((b1, b2), (b2, b1)):
            p0 = k * other - 3 * d
            p1 = k * other + 3 * d
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
    if isinstance(bits, (int, np.integer)):      # bit_grid(stage, 11)
        bits = [int(c) for c in format(int(bits), "b")]
    bits = [int(float(b)) for b in np.asarray(bits).ravel() if str(b).strip()]
    bits = bits[:16] + [0] * max(0, 16 - len(bits))
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
    f = _field(f)
    # Inside the picture area, clear of the title and caption.
    starts = np.column_stack([rng.uniform(-4.5, 4.5, n), rng.uniform(-2.3, 2.3, n)])
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



# -- classic pictures ----------------------------------------------------------
# Pictures the short evaluation asked for and the kit could not draw: the
# coder stacked squares inside one another for "squares on the sides", drew
# a caption for "the angles add up to 180", and invented count_to for
# binary counting.

def squares_on_sides(stage: Stage, a: float = 3, b: float = 4, where: str = "center"):
    """A right triangle with a square on each side: a² and b² fill c²."""
    c = float(np.hypot(a, b))
    P0, P1, P2 = np.array([0., 0, 0]), np.array([a, 0., 0]), np.array([0., b, 0])
    n = np.array([b, a, 0.])                          # outward from the hypotenuse
    tri = Polygon(P0, P1, P2, color=WHITE, fill_opacity=0.2)
    sq_a = Polygon(P0, P1, P1 + [0, -a, 0], P0 + [0, -a, 0], color=BLUE, fill_opacity=0.45)
    sq_b = Polygon(P0, P0 + [-b, 0, 0], P2 + [-b, 0, 0], P2, color=GREEN, fill_opacity=0.45)
    sq_c = Polygon(P1, P2, P2 + n, P1 + n, color=YELLOW, fill_opacity=0.35)
    num = lambda v: f"{v:g}"
    tags = [MathTex(f"a^2 = {num(a * a)}"), MathTex(f"b^2 = {num(b * b)}"),
            MathTex(f"c^2 = {num(round(c * c, 2))}")]
    grp = VGroup(tri, sq_a, sq_b, sq_c)
    stage.place(grp, where)
    _, _, rw, rh = _REGIONS[where]
    if grp.width < 0.6 * rw and grp.height < 0.6 * rh:   # (1, 1) came out tiny
        grp.scale(min(0.8 * rw / grp.width, 0.8 * rh / grp.height))
    k = grp.width / (a + b + max(a, b) + 1e-9)
    for t, sq in zip(tags, (sq_a, sq_b, sq_c)):
        t.scale(min(1.0, max(0.45, 2.2 * k))).move_to(sq.get_center())
    stage.scene.play(Create(tri), run_time=0.8)
    for sq, t in ((sq_a, tags[0]), (sq_b, tags[1]), (sq_c, tags[2])):
        stage.scene.play(GrowFromCenter(sq), FadeIn(t), run_time=0.8)
    stage.scene.play(Indicate(sq_a), Indicate(sq_b), run_time=0.8)
    stage.scene.play(Indicate(sq_c), run_time=0.8)
    out = VGroup(grp, *tags)
    out.squares, out.triangle = [sq_a, sq_b, sq_c], tri
    stage._objects.append(out)
    return out


def angle_sum(stage: Stage, points=((-2.6, -1.3), (2.6, -1.3), (0.9, 1.7)),
              where: str = "center"):
    """A triangle's three angles lifted off and laid side by side on a
    straight line: together they make a half turn, 180°."""
    try:
        V = [_xy(q) for q in points]
    except (TypeError, ValueError, IndexError):
        V = []
    if len(V) != 3:
        V = [_xy(q) for q in ((-2.6, -1.3), (2.6, -1.3), (0.9, 1.7))]
    tri = Polygon(*V, color=WHITE)
    wedges, spans, colors = [], [], (RED, GREEN, BLUE)
    for i in range(3):
        v, p, q = V[i], V[(i + 1) % 3], V[(i + 2) % 3]
        a1 = np.arctan2(*(p - v)[1::-1])
        a2 = np.arctan2(*(q - v)[1::-1])
        span = (a2 - a1) % TAU
        start = a1
        if span > PI:
            start, span = a2, TAU - span
        wedges.append(Sector(arc_center=v, radius=0.6, start_angle=start, angle=span,
                             color=colors[i], fill_opacity=0.75, stroke_width=0))
        spans.append(span)
    grp = VGroup(tri, *wedges)
    stage.place(grp, where)
    stage.scene.play(Create(tri), run_time=0.8)
    stage.scene.play(LaggedStart(*[FadeIn(w) for w in wedges], lag_ratio=0.3), run_time=1.0)
    base = tri.get_bottom() + 0.9 * DOWN
    line = Line(base + 2 * LEFT, base + 2 * RIGHT, color=GREY_B)
    stage.scene.play(Create(line), run_time=0.5)
    moved, t0 = [], 0.0
    for w, span in zip(wedges, spans):
        target = Sector(arc_center=base, radius=0.9, start_angle=t0, angle=span,
                        color=w.get_fill_color(), fill_opacity=0.75, stroke_width=0)
        c = w.copy()
        stage.scene.play(Transform(c, target), run_time=0.9)
        moved.append(c)
        t0 += span
    tag = MathTex(r"180^\circ", font_size=40).next_to(base, DOWN, buff=0.15)
    stage.scene.play(FadeIn(tag), run_time=0.5)
    out = VGroup(grp, line, *moved, tag)
    stage._objects.append(out)
    return out


def count_binary(stage: Stage, bits: int = 4, upto: int | None = None,
                 where: str = "center", step_time: float = 0.45):
    """Bits under their place values (8 4 2 1) counting up, the decimal
    value alongside."""
    bits = int(max(1, min(bits, 8)))
    upto = min(2 ** bits - 1, 15 if upto is None else int(upto))
    cells = VGroup(*[Square(0.9, color=GREY_B) for _ in range(bits)]).arrange(RIGHT, buff=0)
    places = VGroup(*[MathTex(str(2 ** (bits - 1 - i)), font_size=28, color=GREY_B)
                      .next_to(cells[i], UP, buff=0.15) for i in range(bits)])
    digits = VGroup(*[MathTex("0", font_size=44).move_to(cells[i]) for i in range(bits)])
    value = Integer(0, font_size=56, color=YELLOW)
    eq = VGroup(MathTex("=", font_size=48), value).arrange(RIGHT).next_to(cells, RIGHT, buff=0.5)
    grp = VGroup(cells, places, digits, eq)
    stage.place(grp, where)
    stage.scene.play(Create(cells), FadeIn(places), FadeIn(digits), FadeIn(eq), run_time=1.0)
    shown = ["0"] * bits
    for n in range(1, upto + 1):
        s = format(n, f"0{bits}b")
        anims = []
        for i, ch in enumerate(s):
            if shown[i] != ch:
                shown[i] = ch
                new = MathTex(ch, font_size=44).move_to(cells[i])
                anims.append(Transform(digits[i], new))
                anims.append(cells[i].animate.set_fill(BLUE, opacity=0.4 if ch == "1" else 0))
        value.set_value(n)
        stage.scene.play(*anims, run_time=step_time)
    stage._objects.append(grp)
    return grp


def secant_to_tangent(stage: Stage, ax, f, x0: float, h: float = 2.0,
                      run_time: float = 3.0, color=YELLOW):
    """A secant through x0 and x0 + h; h shrinks and the secant becomes the
    tangent, its slope read out (the derivative as a limit)."""
    _need(ax, "c2p", "the axes from draw_axes(stage)", "secant_to_tangent(stage, ax, f, x0)")
    hv = ValueTracker(h)

    def slope():
        d = hv.get_value()
        return (f(x0 + d) - f(x0)) / d

    def line():
        k = slope()
        return Line(ax.c2p(x0 - 1.5, f(x0) - 1.5 * k), ax.c2p(x0 + 1.5 + hv.get_value(),
                    f(x0) + (1.5 + hv.get_value()) * k), color=color)

    sec = always_redraw(line)
    p = Dot(ax.c2p(x0, f(x0)), color=color)
    q = always_redraw(lambda: Dot(ax.c2p(x0 + hv.get_value(), f(x0 + hv.get_value())),
                                  color=RED))
    num = DecimalNumber(slope(), num_decimal_places=2, font_size=34)
    read = VGroup(MathTex(r"\text{slope} =", font_size=34), num).arrange(RIGHT)
    _readout_corner(ax, read)
    stage.scene.play(Create(sec), FadeIn(p), FadeIn(q), FadeIn(read), run_time=1.0)
    num.add_updater(lambda m: m.set_value(slope()))
    stage.scene.play(hv.animate.set_value(0.01 if h > 0 else -0.01), run_time=run_time)
    stage._objects += [sec, p, q, read]
    return sec, read


def swing_pendulum(stage: Stage, length: float = 2.4, amplitude: float = 0.5,
                   swings: float = 2.0, period: float = 2.0):
    """A pendulum swinging, its angle traced against time beside it: the
    trace is a cosine (simple harmonic motion)."""
    pivot = np.array([-4.0, 2.0, 0])
    t = ValueTracker(0)
    theta = lambda: amplitude * np.cos(TAU * t.get_value() / period)
    bob_at = lambda: pivot + length * np.array([np.sin(theta()), -np.cos(theta()), 0])
    rod = always_redraw(lambda: Line(pivot, bob_at(), color=GREY_B))
    bob = always_redraw(lambda: Dot(bob_at(), radius=0.16, color=YELLOW))
    total = swings * period
    ax = Axes(x_range=[0, total, period / 2], y_range=[-1.2 * amplitude, 1.2 * amplitude,
              amplitude / 2], x_length=6.5, y_length=3, tips=False).move_to([2.6, -0.2, 0])
    lab = ax.get_axis_labels(MathTex("t"), MathTex(r"\theta"))
    trace = always_redraw(lambda: ax.plot(
        lambda s: amplitude * np.cos(TAU * s / period),
        x_range=[0, max(t.get_value(), 1e-3)], color=YELLOW))
    stage.scene.play(Create(ax), FadeIn(lab), FadeIn(Dot(pivot, radius=0.05)), run_time=0.8)
    stage.scene.add(rod, bob, trace)
    stage.scene.play(t.animate.set_value(total), run_time=total, rate_func=lambda x: x)
    stage._objects += [rod, bob, ax, lab, trace]
    return ax, trace


def sieve_primes(stage: Stage, n: int = 50, where: str = "center"):
    """The sieve of Eratosthenes: each prime's multiples fade out, the
    primes are what is left."""
    n = int(max(10, min(n, 100)))
    cols = 10
    cells = VGroup(*[VGroup(Square(0.62, stroke_width=1, color=GREY_B),
                            Text(str(k), font_size=20)) for k in range(1, n + 1)])
    cells.arrange_in_grid(cols=cols, buff=0.04)
    stage.place(cells, where)
    stage.scene.play(FadeIn(cells), run_time=0.8)
    stage.scene.play(cells[0].animate.set_opacity(0.15), run_time=0.3)
    out = set()
    palette = [RED, ORANGE, GREEN, TEAL, BLUE]
    for i, p in enumerate(q for q in range(2, int(n ** 0.5) + 1)):
        if p in out:
            continue
        col = palette[i % len(palette)]
        multiples = [m for m in range(p * p, n + 1, p) if m not in out]
        out |= set(multiples)
        stage.scene.play(cells[p - 1][0].animate.set_fill(col, opacity=0.6), run_time=0.3)
        if multiples:
            stage.scene.play(*[cells[m - 1].animate.set_opacity(0.15) for m in multiples],
                             run_time=0.6)
    primes = [k for k in range(2, n + 1) if k not in out]
    stage.scene.play(*[cells[k - 1][0].animate.set_fill(YELLOW, opacity=0.5) for k in primes],
                     run_time=0.8)
    stage._objects.append(cells)
    return cells


KIT_API = """\
stage = Stage(self) already exists. Every block plays its own animation and
returns what it made; keep the return value to reuse it in later beats.
Blocks are verbs (draw_plane, plot_graph); keep what they return in short
nouns (p, ax, g, v) and animate those -- never a block's name.
  stage.title(s)  stage.caption(s)   -- replace the title / caption slot
  stage.equation(tex1, tex2, ..., where="right") -- a derivation, each step
      transforming into the next
  stage.label(m, s)  stage.clear(keep=[...])  stage.pause(seconds)
  draw_square(stage, side=2, label=None)  draw_rectangle(stage, width, height)
  draw_circle(stage, radius=1.5)  draw_polygon(stage, [(x, y), ...])
  draw_dots(stage, n=9, cols=None)  draw_arrow(stage, (x0, y0), (x1, y1))
  draw_line(stage, (x0, y0), (x1, y1), dashed=False)
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
  squares_on_sides(stage, a=3, b=4)      -- squares on a right triangle's sides
  angle_sum(stage)                       -- a triangle's angles laid on a line: 180°
  count_binary(stage, bits=4)            -- bits count up under 8 4 2 1
  secant_to_tangent(stage, ax, f, x0)    -- secant shrinks to the tangent
  swing_pendulum(stage)                  -- pendulum, its angle traced as a cosine
  sieve_primes(stage, n=50)              -- multiples fall away, primes remain
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
             "euler_circle", "squares_on_sides", "angle_sum", "count_binary",
             "secant_to_tangent", "swing_pendulum", "sieve_primes"}


#: Blocks that only set the stage: a grid, axes, a number line. On their own
#: they are a background, not a picture -- GRPO found that drawing a bare
#: plane earned the "draws something" bonus and filled scenes with empty
#: grids. A beat counts as visual only with a block outside this set.
KIT_SCAFFOLD = {"draw_plane", "draw_axes", "draw_number_line",
                "draw_complex_plane"}


# -- tolerance: how models get a block call slightly wrong ---------------------
# One bad call in a scene's first beat used to cost the whole scene: the beat
# was dropped, and every later beat lost the axes it built. These are the
# near-misses a model makes, each turned into the call it meant:
#   * the stage left out -- plot_graph(ax, f) -- when a Stage exists;
#   * a keyword the block does not take (color= on a block without one) --
#     dropped, not a TypeError; extra positional arguments likewise;
#   * a function written as a string -- "x**2", "np.sin(x)", "y = x^2",
#     "lambda x: ..." -- compiled into one.
import functools as _functools
import inspect as _inspect
import math as _math
import re as _re

_FN_SLOTS = {"f", "g", "term", "f0"}
# second-argument slots: the attribute that proves the right thing is there,
# and the block that makes a default one
_MOB_PARAMS = {"m", "secs", "bars_", "arrow", "cells"}  # take a picture
def _on_screen(st, m) -> bool:
    """Is m (or, for a block's tuple result, any part of it) on screen?"""
    parts = m if isinstance(m, (tuple, list)) else [m]
    live = {id(y) for t in st.scene.mobjects for y in t.get_family()}
    return any(id(p) in live for x in parts if hasattr(x, "get_family")
               for p in x.get_family())


_MAKERS = {"draw_plane", "draw_axes", "draw_complex_plane", "draw_number_line"}
# Pictures that bring their own frame: drawn over a plane or axes left on
# screen they are clutter (kit v6 laid angle_sum's triangle over a grid).
_STANDALONE = {"angle_sum", "squares_on_sides", "count_binary", "sieve_primes",
               "swing_pendulum", "circle_to_sine", "draw_dice_grid", "bayes_square",
               "draw_neural_net", "draw_network", "flip_coins", "grow_histogram",
               "hanoi_moves", "bit_grid", "prime_spiral", "fill_halving_squares",
               "slice_circle", "draw_right_triangle", "wind_signal",
               "show_partial_sums", "draw_bars", "draw_array", "convolve_bars"}
_SLOTS = {"plane_": ("c2p", "draw_plane"), "ax": ("c2p", "draw_axes"),
          "cp": ("n2p", "draw_complex_plane"), "nl": ("n2p", "draw_number_line")}

# Axes and NumberPlane have c2p; models also call n2p (NumberLine's and
# ComplexPlane's) on them -- 22 runtime drops in one evaluation round.
from manim import Axes as _Axes


def _axes_n2p(self, x, *rest):
    if isinstance(x, complex):
        return self.c2p(x.real, x.imag)
    return self.c2p(x, rest[0] if rest else 0)


if not hasattr(_Axes, "n2p"):
    _Axes.n2p = _axes_n2p

# Curves sampled at the axis tick step: Manim's plot() takes the axis's step
# when x_range has no third number, so sin(10x) on unit ticks came out as a
# jagged zigzag (teacher batch 11). About 200 samples across the range.
if not getattr(_Axes, "_forge_plot_ok", False):
    _plot = _Axes.plot

    def _fine_plot(self, function, x_range=None, *args, **kw):
        xr = list(x_range) if x_range is not None else [self.x_range[0], self.x_range[1]]
        if len(xr) < 3:
            span = abs(xr[1] - xr[0]) or 1.0
            xr = [xr[0], xr[1], span / 200]
        return _plot(self, function, xr, *args, **kw)
    _Axes.plot = _fine_plot
    _Axes._forge_plot_ok = True

# ax.c2p((x, y)) -- one point, not two numbers -- failed with "setting an
# array element with a sequence" (three runtime drops in one run).
if not getattr(_Axes, "_forge_c2p_ok", False):
    _c2p = _Axes.coords_to_point

    def _coords_to_point(self, *coords, **kw):
        if len(coords) == 1 and np.ndim(coords[0]) == 1 and len(coords[0]) in (2, 3):
            coords = tuple(coords[0][:2])
        return _c2p(self, *coords, **kw)
    _Axes.coords_to_point = _coords_to_point
    _Axes.c2p = _coords_to_point
    _Axes._forge_c2p_ok = True

# A LaTeX string that does not compile (models write \frac{a}{b with a brace
# missing, or unicode) raised and cost its beat; it becomes plain text.
_ManimMathTex, _ManimTex = MathTex, Tex


def _tex_or_text(cls, args, kw):
    try:
        return cls(*args, **kw)
    except (ValueError, RuntimeError, OSError) as e:
        if not any(w in str(e).lower() for w in ("latex", "dvi", "tex")):
            raise
        raw = " ".join(str(a) for a in args)
        raw = _re.sub(r"\\[A-Za-z]+", " ", raw).replace("$", "")
        raw = " ".join(_re.sub(r"[{}]", "", raw).split()) or "?"
        return Text(raw, font_size=kw.get("font_size", 36), color=kw.get("color", WHITE))


def MathTex(*args, **kw):  # noqa: N802 -- stands in for manim's MathTex
    return _tex_or_text(_ManimMathTex, args, kw)


def Tex(*args, **kw):  # noqa: N802
    return _tex_or_text(_ManimTex, args, kw)


def _as_function(v):
    if hasattr(v, "underlying_function"):      # a plotted graph for its function
        return v.underlying_function
    if not isinstance(v, str):
        return v
    src = _re.sub(r"^\s*(?:y|f\s*\(\s*\w\s*\))\s*=\s*", "", v.strip())
    src = src.replace("^", "**")
    if not src.startswith("lambda"):
        var = next((c for c in ("x", "n", "k", "t") if _re.search(rf"\b{c}\b", src)), "x")
        src = f"lambda {var}: {src}"
    env = {k: getattr(np, k) for k in ("sin", "cos", "tan", "exp", "log", "sqrt",
                                          "abs", "pi", "e", "arctan", "sinh",
                                          "cosh", "tanh")}
    env.update(np=np, math=_math)
    try:
        fn = eval(src, env)
        fn(1.0)
        return fn
    except Exception:
        return v


def _tolerant(f):
    sig = _inspect.signature(f)
    params = list(sig.parameters.values())
    names = {q.name for q in params}
    any_kw = any(q.kind is q.VAR_KEYWORD for q in params)
    any_pos = any(q.kind is q.VAR_POSITIONAL for q in params)
    n_pos = sum(q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD) for q in params)

    pos_names = [q.name for q in params
                 if q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD)]
    slot = pos_names[1] if len(pos_names) > 1 and pos_names[1] in _SLOTS else None

    @_functools.wraps(f)
    def call(*args, **kw):
        if (not args or not isinstance(args[0], Stage)) and "stage" not in kw \
                and _CURRENT:
            args = (_CURRENT[0],) + args
        st = args[0] if args and isinstance(args[0], Stage) else None
        # The plane / axes slot left out or filled with something else
        # (apply_matrix(stage, matrix)): the newest one of that kind, or a
        # fresh default one if the scene has none yet.
        if st is not None and slot and slot not in kw and \
                (len(args) < 2 or not hasattr(args[1], _SLOTS[slot][0])):
            have = st._last.get(slot)
            if have is None:
                have = globals()[_SLOTS[slot][1]](st)
            # a wrong picture in the slot is replaced; data is shifted along
            wrong = len(args) > 1 and hasattr(args[1], "get_center")
            args = (args[0], have) + args[2 if wrong else 1:]
        # A plane or axes handed to a block that takes none -- the habit of
        # block(stage, p, ...) carried over: kit v7 called angle_sum,
        # squares_on_sides and swing_pendulum that way every time.
        if st is not None and not slot and len(args) > 1 and \
                pos_names[1:2] and pos_names[1] not in _MOB_PARAMS and \
                (hasattr(args[1], "c2p") or hasattr(args[1], "n2p")):
            args = (args[0],) + args[2:]
        if not any_kw:
            kw = {k: v for k, v in kw.items() if k in names}
        kw = {k: v for k, v in kw.items() if k not in pos_names[:len(args)]}
        # A new plane or axes over one still on screen: two coordinate
        # systems drawn on top of each other (GRPO v3's waves scene). A new
        # coordinate system is a new picture, so the old one is cleared.
        if st is not None and (f.__name__ in _MAKERS or f.__name__ in _STANDALONE):
            old = [m for k, m in st._last.items()
                   if (k in _SLOTS or k == "standalone") and m is not None
                   and _on_screen(st, m)]
            if old:
                st.clear()
                st._last = {k: v for k, v in st._last.items()
                            if k not in _SLOTS and k != "standalone"}
        if not any_pos and len(args) > n_pos:
            args = args[:n_pos]
        try:
            bound = sig.bind_partial(*args, **kw)
        except TypeError:
            return f(*args, **kw)
        for k in _FN_SLOTS & set(bound.arguments):
            bound.arguments[k] = _as_function(bound.arguments[k])
        out = f(*bound.args, **bound.kwargs)
        if st is not None:
            for key, (attr, maker) in _SLOTS.items():
                if f.__name__ == maker:
                    st._last[key] = out
            if f.__name__ == "plot_graph":
                st._last["graph"] = out
            if f.__name__ in _STANDALONE:
                st._last["standalone"] = out
        return out
    call._kit_block = True
    return call


for _n in sorted(KIT_BLOCKS | {"highlight", "pulse"}):
    globals()[_n] = _tolerant(globals()[_n])
del _n
