
        stage = Stage(self)
        # beat 1: The complex plane with the point 2 + i marked
        stage.title("What does multiplying by i do to 2 + i?")
        cp = draw_complex_plane(stage)
        point = mark_point(stage, cp, (2, 1), label="2 + i")
        stage.mark(min_seconds=6.65)

        # beat 2: The plane turns a quarter turn counter-clockwise, landing 2 + i at -1 + 2i
        multiplied_point = multiply_complex(stage, cp, 1j, points=(2 + 1j,))
        stage.caption("× i: a quarter turn")
        point2 = mark_point(stage, cp, (-1, 2), label="-1 + 2i", color=YELLOW)
        stage.mark(min_seconds=8.95)

        # beat 3: The result is a 180-degree flip relative to the start, revealing that i squared is -1
        multiplied_point_2 = multiply_complex(stage, cp, 1j, points=(-1 + 2j,))
        stage.equation(r"i^2 = -1")
        stage.mark(min_seconds=6.02)
        self.wait(1)
