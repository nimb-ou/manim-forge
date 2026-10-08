
        stage = Stage(self)
        # beat 1: Two bars: one cut in quarters with 3 shaded, one cut in sixths with 1 shaded
        stage.title("Adding 3/4 and 1/6")
        w = 6
        x0 = -6
        quarters = VGroup(*[Rectangle(width=w / 4, height=0.6, color=BLUE, fill_color=BLUE, fill_opacity=0.7 if i < 3 else 0) for i in range(4)]).arrange(RIGHT, buff=0).move_to([x0 + w / 2, 1.6, 0])
        sixths = VGroup(*[Rectangle(width=w / 6, height=0.6, color=ORANGE, fill_color=ORANGE, fill_opacity=0.7 if i < 1 else 0) for i in range(6)]).arrange(RIGHT, buff=0).move_to([x0 + w / 2, 0.4, 0])
        stage.play(FadeIn(quarters), FadeIn(sixths))
        stage.add(quarters, sixths)
        stage.label(quarters, "3/4", direction=RIGHT, color=BLUE)
        stage.label(sixths, "1/6", direction=RIGHT, color=ORANGE)
        stage.mark(min_seconds=6.95)

        # beat 2: Both bars re-cut into twelfths: 3/4 becomes 9/12, 1/6 becomes 2/12
        cuts = VGroup(*[Line([x0 + k * w / 12, y - 0.3, 0], [x0 + k * w / 12, y + 0.3, 0], color=GREY_B, stroke_width=1) for k in range(1, 13) for y in (1.6, 0.4)])
        stage.play(Create(cuts))
        stage.add(cuts)
        stage.play(FadeIn(Text("9/12", font_size=26, color=BLUE).move_to([x0 + w / 2, 2.2, 0])), FadeIn(Text("2/12", font_size=26, color=ORANGE).move_to([x0 + w / 2, -0.2, 0])))
        stage.equation(r"\frac{3}{4} = \frac{9}{12}", r"\frac{1}{6} = \frac{2}{12}")
        stage.mark(min_seconds=8.10)

        # beat 3: One bar of twelve twelfths with 9 blue and 2 orange shaded: 11/12
        twelfths = VGroup(*[Rectangle(width=w / 12, height=0.6, color=WHITE, fill_color=BLUE if i < 9 else ORANGE, fill_opacity=0.7 if i < 11 else 0) for i in range(12)]).arrange(RIGHT, buff=0).move_to([x0 + w / 2, -1.6, 0])
        stage.play(FadeIn(twelfths))
        stage.add(twelfths)
        stage.label(twelfths, "11/12", direction=RIGHT, color=GREEN)
        stage.equation(r"\frac{9}{12} + \frac{2}{12}", r"= \frac{11}{12}")
        stage.mark(min_seconds=6.72)
        self.wait(1)
