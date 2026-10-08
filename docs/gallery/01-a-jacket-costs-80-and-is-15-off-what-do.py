
        stage = Stage(self)
        # beat 1: A bar for the £80 price
        stage.title("Discounted Price")
        u = 0.08
        x0 = -6
        b1 = Rectangle(width=80 * u, height=0.6, color=BLUE, fill_opacity=0.6).move_to([x0 + 40 * u, 1.2, 0])
        stage.play(GrowFromEdge(b1, LEFT))
        stage.add(b1)
        stage.play(FadeIn(Text("£80", font_size=28).move_to(b1)))
        stage.mark(min_seconds=2.68)

        # beat 2: 15% cut off: £12 removed in red, £68 left
        b2 = Rectangle(width=68 * u, height=0.6, color=BLUE, fill_opacity=0.6).move_to([x0 + 34 * u, 0, 0])
        cut2 = Rectangle(width=12 * u, height=0.6, color=RED, fill_opacity=0.4).next_to(b2, RIGHT, buff=0)
        stage.play(FadeIn(b2), FadeIn(cut2))
        stage.add(b2, cut2)
        stage.add(Text("£68", font_size=28).move_to(b2))
        stage.label(cut2, "− £12", direction=RIGHT, color=RED)
        stage.mark(min_seconds=4.65)

        # beat 3: The bars beside 80 × 0.85 = 68
        stage.equation(r"80 \times 0.85 = 68")
        stage.mark(min_seconds=3.55)
        self.wait(1)
