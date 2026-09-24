import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from manim import *
from forge.kit.kit import *
import numpy as np


class Gallery3(Scene):
    def construct(self):
        stage = Stage(self)
        stage.title("Waves add")
        ax = axes(stage, x_range=(0, 8), y_range=(-2, 2))
        superpose(stage, ax, [(1, 2, 2), (1, 2, -2)], t_end=3)
        stage.clear()
        stage.title("A square wave from sines")
        ax = axes(stage, x_range=(0, 7), y_range=(-2, 2))
        fourier_series(stage, ax, n_terms=6, run_time=0.6)
        stage.clear()
        stage.title("Halves fill the square")
        halving_squares(stage, n=6)
        stage.clear()
        stage.title("Sorting")
        b = array_bars(stage, [5, 2, 8, 3, 6])
        swap(stage, b, 0, 1)
        stage.clear()
        stage.title("Coin flips")
        coin_flips(stage, n=20)
        stage.clear()
        stage.title("Limits")
        ax = axes(stage, x_range=(0, 8), y_range=(0, 3))
        f = lambda x: 1 + 1 / (x + 0.5)
        g = graph(stage, ax, f)
        epsilon_band(stage, ax, g, 1.0)
        stage.pause(1)
