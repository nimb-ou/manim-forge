import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from manim import *
from forge.kit.kit import *


class Gallery(Scene):
    def construct(self):
        stage = Stage(self)
        stage.title("Linear maps move the grid")
        p = draw_plane(stage)
        i, j = draw_basis(stage, p)
        v = draw_vector(stage, p, (2, 1), label="v")
        stage.caption("A shear keeps i-hat and slides j-hat")
        apply_matrix(stage, p, [[1, 1], [0, 1]], riders=[i, j, v])
        stage.clear()
        stage.title("Slope is a limit")
        ax = draw_axes(stage, x_range=(-1, 4), y_range=(-1, 9))
        f = lambda x: x ** 2
        g = plot_graph(stage, ax, f, label="x^2")
        slide_tangent(stage, ax, f, 0.2, 2.5)
        stage.clear()
        stage.title("Area under a curve")
        ax = draw_axes(stage, x_range=(0, 3), y_range=(0, 9))
        g = plot_graph(stage, ax, f)
        riemann_refine(stage, ax, g, 0, 2.5)
        stage.caption("More strips, less error")
        stage.clear()
        stage.title("Why the area is pi r squared")
        s = slice_circle(stage, n=16)
        unroll_slices(stage, s)
        stage.equation(r"A = \pi r \cdot r", r"A = \pi r^2")
        stage.clear()
        stage.title("Partial sums")
        show_partial_sums(stage, lambda k: 1 / k ** 2, n=10)
        stage.pause(1)
