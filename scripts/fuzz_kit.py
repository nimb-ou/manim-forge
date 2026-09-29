"""Call every kit block the ways a model plausibly would, and render.

A block that raises on a reasonable call costs a whole beat -- in beat 1, a
whole scene. `slide_tangent` failed on any tangent that started on a falling
stretch (a minus sign appearing mid-FadeIn), which no gallery caught because
the galleries only use the arguments I thought of. This renders each block
under several argument styles -- negative ranges, 3-vectors, lists for
tuples, numpy arrays, strings for functions, keywords it does not take --
one scene per block, each call in its own try so one failure does not hide
the next, and reports every call that raised.

    ./.venv/bin/python scripts/fuzz_kit.py [--only slide_tangent,draw_bars]
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forge.app.twostage import kit_source  # noqa: E402

AX = "ax = draw_axes(stage, x_range=(-4, 4), y_range=(-3, 5))\n"
P = "p = draw_plane(stage)\n"
CP = "cp = draw_complex_plane(stage)\n"
NL = "nl = draw_number_line(stage, x_range=(-5, 5))\n"

# block -> (set-up, [calls])
CASES: dict[str, tuple[str, list[str]]] = {
    "draw_square": ("", ["draw_square(stage)", "draw_square(stage, 1.5, 'left', color=RED)",
                         "draw_square(stage, side=3, where=(2, 1))"]),
    "draw_rectangle": ("", ["draw_rectangle(stage, 4, 1)", "draw_rectangle(stage, width=2, height=3, where='right')"]),
    "draw_circle": ("", ["draw_circle(stage)", "draw_circle(stage, 2, 'left', color=GREEN)",
                         "draw_circle(stage, radius=0.5, where=RIGHT)"]),
    "draw_polygon": ("", ["draw_polygon(stage, [(0, 0), (2, 0), (1, 2)])",
                          "draw_polygon(stage, [[0, 0, 0], [2, 0, 0], [1, 2, 0]])",
                          "draw_polygon(stage, np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))"]),
    "draw_dots": ("", ["draw_dots(stage)", "draw_dots(stage, 12, 4)", "draw_dots(stage, n=1)",
                       "draw_dots(stage, 30, cols=10)"]),
    "draw_arrow": ("", ["draw_arrow(stage, (0, 0), (2, 1))", "draw_arrow(stage, ORIGIN, 2 * UP, label='v')",
                        "draw_arrow(stage, [-1, -1, 0], [1, 1, 0], color=RED)"]),
    "draw_line": ("", ["draw_line(stage, (0, 0), (3, 0))", "draw_line(stage, LEFT, RIGHT, dashed=True)"]),
    "draw_plane": ("", ["draw_plane(stage)", "draw_plane(stage, 'left')", "draw_plane(stage, x_extent=6, y_extent=4)",
                        "draw_plane(stage, where='right', x_extent=3)"]),
    "draw_vector": (P, ["draw_vector(stage, p, (2, 1))", "draw_vector(stage, p, (-3, -2), label='v')",
                        "draw_vector(stage, p, [1, 2, 0], color=RED)", "draw_vector(stage, p, np.array([0, -1]))",
                        "draw_vector(stage, p, (0, 0))"]),
    "draw_basis": (P, ["draw_basis(stage, p)"]),
    "apply_matrix": (P + "v = draw_vector(stage, p, (1, 1))\n",
                     ["apply_matrix(stage, p, [[0, -1], [1, 0]])", "apply_matrix(stage, p, [[2, 1], [1, 3]], riders=[v])",
                      "apply_matrix(stage, p, np.array([[1, 1], [0, 1]]))", "apply_matrix(stage, p, ((1, 0), (0, -1)), [v])",
                      "apply_matrix(stage, p, [[0, 0], [0, 0]])"]),
    "draw_unit_square": (P, ["draw_unit_square(stage, p)", "draw_unit_square(stage, p, color=GREEN)"]),
    "draw_span": (P, ["draw_span(stage, p, (1, 2))", "draw_span(stage, p, (-2, 1))", "draw_span(stage, p, (0, 0))"]),
    "scale_vector": (P + "v = draw_vector(stage, p, (1, 1))\n",
                     ["scale_vector(stage, p, v, 2)", "scale_vector(stage, p, v, -1.5)", "scale_vector(stage, p, v, 0)"]),
    "draw_axes": ("", ["draw_axes(stage)", "draw_axes(stage, (-5, 5), (-2, 2))", "draw_axes(stage, x_range=[0, 10, 2], y_range=[0, 100, 20])",
                       "draw_axes(stage, x_range=(-3.14, 3.14), y_range=(-1.5, 1.5), where='left')",
                       "draw_axes(stage, x_range=(0, 1000), y_range=(0, 1))"]),
    "plot_graph": (AX, ["plot_graph(stage, ax, lambda x: x**2 / 4)", "plot_graph(stage, ax, np.sin, label='sin x')",
                        "plot_graph(stage, ax, lambda x: 1 / x)", "plot_graph(stage, ax, lambda x: np.exp(x))",
                        "plot_graph(stage, ax, lambda x: x, x_range=(-2, 2), color=RED)",
                        "plot_graph(stage, ax, 'x**3 - x')", "plot_graph(stage, ax, lambda x: np.sqrt(x))",
                        "plot_graph(stage, ax, lambda x: np.log(x))", "plot_graph(stage, ax, lambda x: np.tan(x))"]),
    "slide_tangent": (AX, ["slide_tangent(stage, ax, lambda x: x**2 / 4, -3, 3)",
                           "slide_tangent(stage, ax, lambda x: x**2, 2, -2)", "slide_tangent(stage, ax, np.sin, 0, 3)",
                           "slide_tangent(stage, ax, lambda x: -x**3 / 8, -2, 2)",
                           "slide_tangent(stage, ax, lambda x: 100 * x, 0, 1)"]),
    "shade_area": (AX, ["shade_area(stage, ax, lambda x: x**2 / 4, 0, 2)", "shade_area(stage, ax, np.sin, -3, 3)",
                        "shade_area(stage, ax, lambda x: x, 2, 0)"]),
    "riemann_refine": (AX, ["riemann_refine(stage, ax, lambda x: x**2 / 4, 0, 3)",
                            "riemann_refine(stage, ax, np.sin, -3, 3, ns=(2, 4))",
                            "riemann_refine(stage, ax, lambda x: -x, 0, 2, ns=[4, 8, 16, 32, 64])"]),
    "trace_graph": (AX, ["trace_graph(stage, ax, lambda x: x**2 / 4, -3, 3)", "trace_graph(stage, ax, np.cos, 3, -3)"]),
    "draw_number_line": ("", ["draw_number_line(stage)", "draw_number_line(stage, (-10, 10))",
                              "draw_number_line(stage, x_range=[0, 1, 0.25])", "draw_number_line(stage, (0, 100))"]),
    "mark_point": (NL, ["mark_point(stage, nl, 2)", "mark_point(stage, nl, -3.5, label='a')", "mark_point(stage, nl, 99)"]),
    "draw_bars": ("", ["draw_bars(stage, [3, 1, 4, 1, 5])", "draw_bars(stage, [0.2, 0.5, 0.3], labels=['a', 'b', 'c'])",
                       "draw_bars(stage, [-1, 2, -3])", "draw_bars(stage, [0, 0, 0])", "draw_bars(stage, np.array([1, 2]))",
                       "draw_bars(stage, [5])", "draw_bars(stage, list(range(40)))"]),
    "show_partial_sums": ("", ["show_partial_sums(stage, lambda k: 1 / 2**k)", "show_partial_sums(stage, lambda k: 1 / k, n=8)",
                               "show_partial_sums(stage, lambda n: (-1)**n / (n + 1), 12)",
                               "show_partial_sums(stage, lambda k: k, 5)", "show_partial_sums(stage, '1/2**n', 6)"]),
    "slice_circle": ("", ["slice_circle(stage)", "slice_circle(stage, 4)", "slice_circle(stage, n=40, r=1, where='center')"]),
    "unroll_slices": ("s = slice_circle(stage, 8)\n", ["unroll_slices(stage, s)"]),
    "draw_right_triangle": ("", ["draw_right_triangle(stage)", "draw_right_triangle(stage, 4, 3)",
                                 "draw_right_triangle(stage, a=1, b=5, where='left')"]),
    "draw_dice_grid": ("", ["draw_dice_grid(stage)", "draw_dice_grid(stage, highlight_sum=7)",
                            "draw_dice_grid(stage, highlight_sum=13)"]),
    "show_determinant": (P, ["show_determinant(stage, p, [[2, 1], [1, 2]])", "show_determinant(stage, p, [[1, 2], [2, 4]])",
                             "show_determinant(stage, p, [[0, 1], [1, 0]])", "show_determinant(stage, p, np.array([[3, 0], [0, 2]]))"]),
    "show_eigenvectors": (P, ["show_eigenvectors(stage, p, [[2, 0], [0, 3]])", "show_eigenvectors(stage, p, [[3, 1], [0, 2]])",
                              "show_eigenvectors(stage, p, [[0, -1], [1, 0]])", "show_eigenvectors(stage, p, [[1, 1], [0, 1]])"]),
    "taylor_approximate": (AX, ["taylor_approximate(stage, ax, np.cos, 0, [1, 0, -1, 0, 1])",
                                "taylor_approximate(stage, ax, np.exp, 0, [1, 1, 1, 1])",
                                "taylor_approximate(stage, ax, np.sin, 1, [0.84, 0.54, -0.84])",
                                "taylor_approximate(stage, ax, lambda x: np.log(1 + x), 0, [0, 1, -1, 2])"]),
    "circle_to_sine": ("", ["circle_to_sine(stage)", "circle_to_sine(stage, turns=2)"]),
    "draw_vector_field": ("", ["draw_vector_field(stage, lambda x, y: (-y, x))", "draw_vector_field(stage, lambda x, y: (x, y))",
                               "draw_vector_field(stage, lambda x, y: np.array([1, 0]))",
                               "draw_vector_field(stage, lambda p: np.array([-p[1], p[0], 0]))"]),
    "draw_complex_plane": ("", ["draw_complex_plane(stage)", "draw_complex_plane(stage, 'left')"]),
    "multiply_complex": (CP, ["multiply_complex(stage, cp, 1j)", "multiply_complex(stage, cp, 1 + 1j)",
                              "multiply_complex(stage, cp, 2)", "multiply_complex(stage, cp, complex(0, -1), points=[1, 2j])",
                              "multiply_complex(stage, cp, -1)"]),
    "draw_neural_net": ("", ["draw_neural_net(stage)", "draw_neural_net(stage, [2, 3, 1])", "draw_neural_net(stage, layers=(8, 16, 4))",
                             "draw_neural_net(stage, (784, 16, 10))"]),
    "draw_network": ("", ["draw_network(stage, {'A': (0, 0), 'B': (2, 1), 'C': (2, -1)}, [('A', 'B'), ('A', 'C')])",
                          "draw_network(stage, {'A': (-2, 0), 'B': (2, 0)}, [])",
                          "draw_network(stage, {1: (0, 1), 2: (1, 0)}, [(1, 2)])"]),
    "grow_histogram": ("", ["grow_histogram(stage, lambda rng, k: rng.normal(size=k), bins=np.linspace(-3, 3, 13))",
                            "grow_histogram(stage, lambda rng, k: rng.integers(1, 7, size=k), bins=[1, 2, 3, 4, 5, 6, 7])",
                            "grow_histogram(stage, lambda rng, k: rng.uniform(0, 1, k), bins=10)"]),
    "animate_wave": (AX, ["animate_wave(stage, ax)", "animate_wave(stage, ax, amp=2, k=1, omega=3)"]),
    "superpose_waves": (AX, ["superpose_waves(stage, ax, [(1, 1, 1), (0.5, 3, 2)])", "superpose_waves(stage, ax, [(1, 2, 2)])",
                             "superpose_waves(stage, ax, [(1, 1, 1), (1, 1, -1)])"]),
    "build_fourier_series": (AX, ["build_fourier_series(stage, ax)", "build_fourier_series(stage, ax, n_terms=3)",
                                  "build_fourier_series(stage, ax, 15)"]),
    "fill_halving_squares": ("", ["fill_halving_squares(stage)", "fill_halving_squares(stage, 3)", "fill_halving_squares(stage, n=12)"]),
    "narrow_epsilon_band": (AX, ["narrow_epsilon_band(stage, ax, lambda x: 1 + 1 / (x + 5), 1)",
                                 "narrow_epsilon_band(stage, ax, lambda x: np.sin(x) / x, 1, eps=0.3)",
                                 "narrow_epsilon_band(stage, ax, lambda x: 2, 2)"]),
    "draw_array": ("", ["draw_array(stage, [5, 2, 8, 1, 9])", "draw_array(stage, [1])", "draw_array(stage, [3, -1, 0])",
                        "draw_array(stage, list(range(20)))"]),
    "swap_bars": ("b = draw_array(stage, [5, 2, 8, 1])\n", ["swap_bars(stage, b, 0, 1)", "swap_bars(stage, b, 3, 0)",
                                                             "swap_bars(stage, b, 2, 2)"]),
    "flip_coins": ("", ["flip_coins(stage)", "flip_coins(stage, 100, 0.3)", "flip_coins(stage, n=5)"]),
    "bayes_square": ("", ["bayes_square(stage)", "bayes_square(stage, 0.5, 0.8, 0.2)", "bayes_square(stage, prior=0.001)",
                          "bayes_square(stage, 0.1, 1.0, 0.0)"]),
    "gradient_descent": (AX, ["gradient_descent(stage, ax, lambda x: x**2 / 4, 3)", "gradient_descent(stage, ax, lambda x: x**2 / 4, -3, lr=0.5)",
                              "gradient_descent(stage, ax, lambda x: (x - 1)**2, 3, lr=0.9, steps=20)",
                              "gradient_descent(stage, ax, np.cos, 0.5)", "gradient_descent(stage, ax, lambda x: x**2, 3, lr=1.5)"]),
    "convolve_bars": ("", ["convolve_bars(stage, [1, 2, 3], [1, 1])", "convolve_bars(stage, [1, 0, -1, 2], [0.5, 0.5, 0.5])",
                           "convolve_bars(stage, [1], [1])"]),
    "project_vector": (P, ["project_vector(stage, p, (2, 1), (1, 0))", "project_vector(stage, p, (-1, 2), (1, 1))",
                           "project_vector(stage, p, (1, 0), (0, 1))"]),
    "basis_grid": (P, ["basis_grid(stage, p, (1, 0), (1, 1))", "basis_grid(stage, p, (2, 1), (-1, 1))"]),
    "wind_signal": ("", ["wind_signal(stage)", "wind_signal(stage, freqs=(2, 5))", "wind_signal(stage, (3,), wind_from=0.2)"]),
    "prime_spiral": ("", ["prime_spiral(stage)", "prime_spiral(stage, 200)", "prime_spiral(stage, n=20000)"]),
    "diffuse_heat": (AX, ["diffuse_heat(stage, ax, lambda x: np.exp(-x**2))", "diffuse_heat(stage, ax, lambda x: 1.0 * (abs(x) < 1))"]),
    "hanoi_moves": ("", ["hanoi_moves(stage)", "hanoi_moves(stage, 1)", "hanoi_moves(stage, n=5)"]),
    "bit_grid": ("", ["bit_grid(stage, [1, 0, 1, 1])", "bit_grid(stage, [[1, 0], [0, 1]])", "bit_grid(stage, '1011')",
                      "bit_grid(stage, [[1, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]], highlight_cols=[1, 3])"]),
    "flow_particles": ("", ["flow_particles(stage, lambda x, y: (-y, x))", "flow_particles(stage, lambda x, y: (1, 0), n=20)"]),
    "euler_circle": (CP, ["euler_circle(stage, cp)", "euler_circle(stage, cp, t_end=PI)"]),
    "highlight": ("s = draw_square(stage)\n", ["highlight(stage, s)", "highlight(s)"]),
    "pulse": ("s = draw_square(stage)\n", ["pulse(stage, s)"]),
    "squares_on_sides": ("", ["squares_on_sides(stage)", "squares_on_sides(stage, 1, 1)",
                              "squares_on_sides(stage, 5, 12, where='left')"]),
    "angle_sum": ("", ["angle_sum(stage)", "angle_sum(stage, [(-2, -1), (2, -1), (-1.5, 2)])"]),
    "count_binary": ("", ["count_binary(stage)", "count_binary(stage, 3)", "count_binary(stage, bits=8, upto=5)"]),
    "secant_to_tangent": (AX, ["secant_to_tangent(stage, ax, lambda x: x**2 / 4, 1)",
                               "secant_to_tangent(stage, ax, np.sin, -1, h=-2)",
                               "secant_to_tangent(stage, ax, lambda x: -x**2 / 4, 2)"]),
    "swing_pendulum": ("", ["swing_pendulum(stage)", "swing_pendulum(stage, amplitude=1.0, swings=1)"]),
    "plot_points": ("ax = draw_axes(stage, x_range=(0, 10), y_range=(0, 10))", [
        "plot_points(stage, ax, [(1, 2), (3, 4), (5, 5), (7, 8)])",
        "plot_points(ax, [(1, 2), (3, 4)])",
        "plot_points(stage, ax, [[1, 2, 3, 4], [2, 3, 5, 7]])",
        "d = plot_points(stage, ax, np.array([[1, 1], [2, 4], [3, 9]]))\nhighlight(stage, d)"]),
    "mark_point_xy": ("ax = draw_axes(stage, x_range=(0, 5), y_range=(0, 64))", [
        "mark_point(stage, ax, 1, 32, label='N = 32')", "mark_point(stage, ax, (2, 16))"]),
    "aliases": ("", ["draw_triangle(stage)", "draw_triangle(stage, [(0, 0), (3, 0), (0, 2)])",
                     "draw_dice(stage)", "p = draw_axes(stage)\ndraw_point(stage, p, 1, 1)",
                     "ax = draw_axes(stage)\ndraw_scatter(stage, ax, [(1, 1), (2, 2)])"]),
    "sieve_primes": ("", ["sieve_primes(stage)", "sieve_primes(stage, 30)", "sieve_primes(stage, n=100)"]),
    "fixes": ("", [
        "ax = draw_axes(stage)\nd = Dot(ax.c2p((1, 2)))",
        "ax = draw_axes(stage, (-3, 3), (-2, 2))\ng = plot_graph(stage, ax, np.cos)\ntaylor_approximate(stage, ax, np.cos, 0)",
        "ax = draw_axes(stage, (-3, 3), (-2, 2))\ntaylor_approximate(stage, ax, np.cos, 0, 4)",
        "ax = draw_axes(stage, (-3, 3), (-1, 9))\ng = plot_graph(stage, ax, lambda x: x**2)\ngradient_descent(stage, ax, g, 2)",
        "bit_grid(stage, 11)", "bit_grid(stage, '1011')",
        "stage.equation(r'\\frac{a}{b')", "m = MathTex(r'\\frac{1}{2')\nstage.add(m)",
        "stage.caption('$x^2 + \\oops{y}$')"]),
    "v7_notes": ("", [
        "p = draw_plane(stage)\napply_matrix(stage, p, [[1, 0], [0, 2]], riders=[(2, 3), (1, 1)])",
        "p = draw_plane(stage)\nshow_determinant(stage, p)",
        "ax = draw_axes(stage, (0, 4), (0, 9))\np = mark_point(stage, ax, (2, 4), label='f(2) = 4')",
        "ax = draw_axes(stage, (0, 4), (0, 9))\ng = plot_graph(stage, ax, lambda x: x**2 / 2)\nsec = secant_to_tangent(stage, ax, g, 1)\nstage.label(sec, 'Slope: 2')",
        "ax = draw_axes(stage, (0, 4), (0, 9))\ng = plot_graph(stage, ax, lambda x: x**2 / 2)\nt = slide_tangent(stage, ax, g, 1)",
        "ax = draw_axes(stage)\nshow_partial_sums(stage, ax, [0.5, 0.25, 0.125, 0.0625])",
        "ax = draw_axes(stage)\nq = stage.coords_to_point(0, 1)",
        "p = draw_plane(stage)\nangle_sum(stage, p)", "p = draw_plane(stage)\nsq = squares_on_sides(stage, p)\nstage.label(sq.squares[0], 'A')",
        "p = draw_plane(stage)\nswing_pendulum(stage, p, length=2, angle=60, swings=0.5)",
        "draw_network(stage, layers=[2, 3, 1])",
        "ax = draw_axes(stage)\nstage.label(ax.c2p(1, 1), 'x0')",
        "poly = Polygon((0, 0), (2, 0), (1, 1))\nstage.add(poly)", "l = Line((0, 0), (2, 1))\nd = Dot((1, 1))\nstage.add(l, d)",
        "g = sieve_primes(stage, 20)\nmark_point(stage, g, 2, label='2')"]),
    "misuse": ("", [
        "p = draw_plane(stage)\napply_matrix(stage, [[0, -1], [1, 0]])",
        "show_determinant(stage, [[2, 1], [1, 2]])", "show_eigenvectors(stage, [[2, 0], [0, 3]])",
        "ax = draw_axes(stage)\nd = Dot(ax.n2p(2))", "p = draw_plane(stage)\nd = Dot(p.n2p(1))",
        "ax = draw_axes(stage, (0, 3), (0, 9))\ng = plot_graph(stage, ax, lambda x: x**2)\nshade_area(stage, ax, g, (0, 2))",
        "ax = draw_axes(stage, (0, 3), (0, 9))\ng = plot_graph(stage, ax, lambda x: x**2)\nshade_area(stage, ax, 0, 2)",
        "ax = draw_axes(stage, (0, 3), (0, 9))\nriemann_refine(stage, ax, lambda x: x**2, x_range=(0, 2))",
        "net = draw_neural_net(stage)\nhighlight(stage, net.layers[1])",
        "draw_right_triangle(stage, 3, 4, a=3)", "draw_vector(stage, (1, 2))",
        "plot_graph(stage, lambda x: x**2 / 4)"]),
    "stage": ("s = draw_circle(stage)\n", [
        "stage.title('Hello')", "stage.caption('A short caption.')", "stage.equation('a^2 + b^2 = c^2')",
        "stage.equation(r'\\frac{1}{2}', where='left')", "stage.label(s, 'r')", "stage.label('r', s)",
        "stage.place(s, 'left')", "stage.clear()", "stage.pause(0.2)", "stage.title('$x^2$')",
        "stage.caption('Area = $\\\\pi r^2$')", "stage.title('f(x) = x^2')"]),
}

HEAD = "from manim import *\nimport numpy as np\n"


def scene(block: str, setup: str, calls: list[str], log: str) -> str:
    """One scene per block; each call's verdict is appended to ``log`` (the
    harness keeps only the tail of stdout)."""
    say = f"open({log!r}, 'a').write"
    body = []
    for i, c in enumerate(calls):
        body.append(f"        try:\n"
                    f"            stage = Stage(self)\n"
                    + "".join(f"            {l}\n" for l in setup.splitlines())
                    + "".join(f"            {l}\n" for l in c.splitlines())
                    + f"            {say}('FUZZ OK {block} {i}\\n')\n"
                    f"        except Exception as e:\n"
                    f"            {say}('FUZZ FAIL {block} {i} ' + type(e).__name__ + ' ' + str(e)[:160].replace(chr(10), ' ') + '\\n')\n"
                    f"        self.clear()\n")
    return (HEAD + kit_source() + "\n\nclass Fuzz(Scene):\n    def construct(self):\n"
            + "".join(body) + "        self.wait(0.1)\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from forge.harness import RenderHarness
    h = RenderHarness(python_bin=str(ROOT / ".venv" / "bin" / "python"),
                      cache_dir=str(ROOT / "data" / "frames"), timeout=900)
    todo = {k: v for k, v in CASES.items()
            if not a.only or k in a.only.split(",")}

    def one(item):
        block, (setup, calls) = item
        log = Path(tempfile.mkdtemp()) / f"{block}.txt"
        r = h.render(scene(block, setup, calls, str(log)), quality="low",
                     frames=1, use_cache=False)
        out = log.read_text() if log.exists() else ""
        fails = re.findall(rf"^FUZZ FAIL {block} (\d+) (.*)$", out, re.M)
        oks = len(re.findall(rf"^FUZZ OK {block} ", out, re.M))
        crash = "" if r.ok else (r.stderr or "")[-300:]
        return block, calls, oks, fails, crash

    bad = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for block, calls, oks, fails, crash in ex.map(one, todo.items()):
            if not fails and oks == len(calls):
                print(f"ok    {block} ({oks})", flush=True)
                continue
            bad += 1
            print(f"FAIL  {block}: {oks}/{len(calls)} ok", flush=True)
            for i, err in fails:
                print(f"        {calls[int(i)]}\n          -> {err}", flush=True)
            if crash and not fails:
                print(f"        scene crashed: {' '.join(crash.split())[-300:]}")
    print(f"{bad} of {len(todo)} blocks with failures")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
