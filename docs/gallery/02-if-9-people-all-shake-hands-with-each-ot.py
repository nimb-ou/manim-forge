
        stage = Stage(self)
        # beat 1: Nine people as dots in a circle, labelled A to I
        stage.title("Handshakes with 9 People")
        names = "ABCDEFGHI"
        nodes = {n: (0.8 * np.cos(PI / 2 - k * TAU / 9), 0.9 * np.sin(PI / 2 - k * TAU / 9)) for k, n in enumerate(names)}
        dots, _ = draw_network(stage, nodes, [])
        stage.mark(min_seconds=4.27)

        # beat 2: A's eight handshakes drawn as lines to everyone else
        pos = {n: d.get_center() for n, d in dots.items()}
        a_lines = [Line(pos["A"], pos[n], color=YELLOW) for n in "BCDEFGHI"]
        stage.play(*[Create(l) for l in a_lines])
        stage.add(*a_lines)
        stage.caption("A: 8 handshakes")
        stage.mark(min_seconds=3.10)

        # beat 3: All 36 lines between the nine people drawn
        rest = [Line(pos[a], pos[b], color=BLUE) for i, a in enumerate(names) for b in names[i + 1:] if a != "A"]
        stage.play(LaggedStart(*[Create(l) for l in rest], lag_ratio=0.1))
        stage.add(*rest)
        stage.caption("8 + 7 + 6 + 5 + 4 + 3 + 2 + 1 = 36")
        stage.mark(min_seconds=7.97)

        # beat 4: The picture beside 9 × 8 ÷ 2 = 36
        stage.equation(r"\frac{9 \times 8}{2} = 36")
        stage.mark(min_seconds=8.35)
        self.wait(1)
