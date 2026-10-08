
        stage = Stage(self)
        # beat 1: A balance with four x-boxes and 3 unit blocks on the left, 19 unit blocks on the right
        stage.title("Solving 4x + 3 = 19")
        beam = Line([-5, 0, 0], [5, 0, 0], color=GREY_B, stroke_width=8)
        pivot = Triangle(color=GREY_B, fill_opacity=0.8).scale(0.4).move_to([0, -0.35, 0])
        stage.play(Create(beam), FadeIn(pivot))
        stage.add(beam, pivot)
        xs = VGroup(*[Square(0.6, color=BLUE, fill_opacity=0.6) for _ in range(4)]).arrange(RIGHT, buff=0.1).move_to([-3.4, 0.4, 0])
        ones_l = VGroup(*[Square(0.28, color=YELLOW, fill_opacity=0.9) for _ in range(3)]).arrange(RIGHT, buff=0.06).move_to([-1.4, 0.25, 0])
        right = VGroup(*[Square(0.28, color=YELLOW, fill_opacity=0.9) for _ in range(19)]).arrange_in_grid(rows=2, cols=10, buff=0.06).move_to([2.6, 0.4, 0])
        stage.play(FadeIn(xs), FadeIn(ones_l), FadeIn(right))
        stage.add(xs, ones_l, right)
        for b in xs:
            t = MathTex("x", font_size=30).move_to(b)
            stage.add(t)
        stage.equation(r"4x + 3 = 19", where="center")
        stage.mark(min_seconds=7.05)

        # beat 2: Three unit blocks removed from each side, leaving four x-boxes against 16
        stage.play(FadeOut(ones_l), *[FadeOut(right[k]) for k in range(16, 19)])
        stage.equation(r"4x + 3 = 19", r"4x = 16")
        stage.mark(min_seconds=5.72)

        # beat 3: The 16 weights split into 4 groups of 4, one per box, beside x = 4
        stage.play(*[right[k].animate.set_color([GREEN, ORANGE, RED, BLUE][k // 4]) for k in range(16)])
        stage.equation(r"4x + 3 = 19", r"4x = 16", r"x = 4")
        stage.mark(min_seconds=5.27)
        self.wait(1)
