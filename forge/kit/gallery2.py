import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from manim import *
from forge.kit.kit import *
import numpy as np


class Gallery2(Scene):
    def construct(self):
        stage = Stage(self)
        stage.title("Determinant is area")
        p = draw_plane(stage)
        show_determinant(stage, p, [[2, 1], [0, 1.5]])
        stage.clear()
        stage.title("Eigenvectors stay on their line")
        p = draw_plane(stage)
        show_eigenvectors(stage, p, [[2, 1], [0, 1]])
        stage.clear()
        stage.title("Taylor polynomials of cos")
        ax = draw_axes(stage, x_range=(-4, 4), y_range=(-2, 2))
        plot_graph(stage, ax, np.cos, color=BLUE)
        taylor_approximate(stage, ax, np.cos, 0, [1, 0, -1, 0, 1, 0, -1])
        stage.clear()
        stage.title("Sine from a circle")
        circle_to_sine(stage)
        stage.clear()
        stage.title("Multiplying by i rotates")
        cp = draw_complex_plane(stage)
        multiply_complex(stage, cp, 1j, points=(2 + 1j,))
        stage.clear()
        stage.title("A neural network")
        draw_neural_net(stage, (3, 5, 4, 2))
        stage.clear()
        stage.title("A field")
        draw_vector_field(stage, lambda x, y: (-y, x))
        stage.clear()
        stage.title("Averages look normal")
        grow_histogram(stage, lambda rng, k: rng.uniform(size=(k, 6)).mean(axis=1),
                        bins=list(np.linspace(0.1, 0.9, 17)))
        stage.pause(1)
