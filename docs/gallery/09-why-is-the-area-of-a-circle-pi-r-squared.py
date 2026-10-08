
        stage = Stage(self)
        # beat 1: A circle of radius 2 cut into 8 alternating slices
        r = 2.0
        n = 8
        secs = slice_circle(stage, n=n)
        stage.title("Why πr²")
        stage.caption(f"Cut a circle of radius {r} into {n} slices")
        stage.pause(1.5)
        stage.mark(min_seconds=5.60)

        # beat 2: The slices laid tip-to-tip to form a rough parallelogram
        rect = unroll_slices(stage, secs)
        stage.caption("Lay them out: a bumpy parallelogram")
        stage.pause(1.0)
        stage.mark(min_seconds=5.95)

        # beat 3: The shape approximates a rectangle with height r and width πr
        stage.clear()
        r = 2.0
        n = 32
        secs = slice_circle(stage, n=n)
        rect = unroll_slices(stage, secs)
        stage.caption(f"More slices ({n}): the shape straightens out")
        stage.pause(0.8)
        stage.mark(min_seconds=8.72)

        # beat 4: Label the height as r and the width as πr
        stage.label(rect, r"{r}", direction=LEFT, color=GREEN)
        stage.label(rect, r"\pi r", direction=DOWN, color=RED)
        stage.equation(r"Height = r, Width = \pi r")
        stage.pause(1.2)
        stage.mark(min_seconds=8.70)

        # beat 5: Calculate the area by multiplying width and height
        stage.equation(r"A = (\pi r) \cdot r = \pi r^2")
        stage.title("Conclusion: Area = πr²")
        stage.caption("Area = width × height = πr × r = πr²")
        stage.pause(1.5)
        stage.mark(min_seconds=7.42)
        self.wait(1)
