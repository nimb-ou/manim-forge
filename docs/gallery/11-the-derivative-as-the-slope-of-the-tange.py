
        stage = Stage(self)
        # beat 1: A hill-shaped curve peaking at x = 2
        stage.title("The derivative as slope")
        ax = draw_axes(stage, x_range=(0, 4), y_range=(0, 5))
        f = lambda x: 4 - (x - 2) ** 2
        g = plot_graph(stage, ax, f)
        stage.mark(min_seconds=6.12)

        # beat 2: A tangent sliding up the hill, its slope falling to zero
        tan = slide_tangent(stage, ax, f, 0.5, 2)
        stage.caption("Climbing: the slope shrinks to zero")
        stage.mark(min_seconds=6.35)

        # beat 3: The tangent sliding down the far side, its slope negative
        stage.clear(keep=[ax, g])
        tan2 = slide_tangent(stage, ax, f, 2, 3.5)
        stage.caption("Past the peak the slope is negative")
        stage.mark(min_seconds=3.93)

        # beat 4: A flat yellow line resting on the peak
        stage.clear(keep=[ax, g])
        top = mark_point(stage, ax, (2, 4), label="top")
        flat = draw_line(stage, ax.c2p(1, 4), ax.c2p(3, 4), color=YELLOW)
        stage.equation(r"f'(2) = 0", where="right")
        stage.mark(min_seconds=5.70)
        self.wait(1)
