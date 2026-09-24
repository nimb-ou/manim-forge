import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from manim import *
from forge.kit.kit import *
import numpy as np


class Gallery2(Scene):
    def construct(self):
        stage = Stage(self)
        stage.title("Determinant is area")
        p = plane(stage)
        determinant(stage, p, [[2, 1], [0, 1.5]])
        stage.clear()
        stage.title("Eigenvectors stay on their line")
        p = plane(stage)
        eigenvectors(stage, p, [[2, 1], [0, 1]])
        stage.clear()
        stage.title("Taylor polynomials of cos")
        ax = axes(stage, x_range=(-4, 4), y_range=(-2, 2))
        graph(stage, ax, np.cos, color=BLUE)
        taylor(stage, ax, np.cos, 0, [1, 0, -1, 0, 1, 0, -1])
        stage.clear()
        stage.title("Sine from a circle")
        unit_circle_wave(stage)
        stage.clear()
        stage.title("Multiplying by i rotates")
        cp = complex_plane(stage)
        multiply_by(stage, cp, 1j, points=(2 + 1j,))
        stage.clear()
        stage.title("A neural network")
        neural_net(stage, (3, 5, 4, 2))
        stage.clear()
        stage.title("A field")
        vector_field(stage, lambda x, y: (-y, x))
        stage.clear()
        stage.title("Averages look normal")
        histogram_grows(stage, lambda rng, k: rng.uniform(size=(k, 6)).mean(axis=1),
                        bins=list(np.linspace(0.1, 0.9, 17)))
        stage.pause(1)
