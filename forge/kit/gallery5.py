import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from manim import *
from forge.kit.kit import *
import numpy as np


class Gallery5(Scene):
    def construct(self):
        stage = Stage(self)
        stage.title("Squares on the sides")
        squares_on_sides(stage, 3, 4)
        stage.clear()
        stage.title("The angles make a straight line")
        angle_sum(stage)
        stage.clear()
        stage.title("Counting in binary")
        count_binary(stage, 4, upto=6)
        stage.clear()
        stage.title("Secant to tangent")
        ax = draw_axes(stage, x_range=(-1, 4), y_range=(-1, 9))
        f = lambda x: x ** 2 / 2
        plot_graph(stage, ax, f)
        secant_to_tangent(stage, ax, f, 1)
        stage.clear()
        stage.title("A pendulum")
        swing_pendulum(stage, swings=1)
        stage.clear()
        stage.title("The sieve")
        sieve_primes(stage, 50)
        self.wait(0.5)
