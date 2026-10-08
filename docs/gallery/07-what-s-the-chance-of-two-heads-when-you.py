
        stage = Stage(self)
        # beat 1: A tree: start splitting into H and T, each splitting again into H and T, four ends HH, HT, TH, TT
        stage.title("Two coin flips")
        nodes = {"start": (-0.9, 0), "H": (-0.2, 0.5), "T": (-0.2, -0.5),
                 "HH": (0.7, 0.8), "HT": (0.7, 0.25), "TH": (0.7, -0.25), "TT": (0.7, -0.8)}
        edges = [("start", "H"), ("start", "T"), ("H", "HH"), ("H", "HT"), ("T", "TH"), ("T", "TT")]
        dots, links = draw_network(stage, nodes, edges)
        stage.mark(min_seconds=7.90)

        # beat 2: The ends HH and TT stay dim, while HT and TH light up green
        for k in ("HH",):
            highlight(stage, dots[k], color=GREEN)
        stage.caption("1 of the 4 outcomes")
        stage.mark(min_seconds=5.45)

        # beat 3: The tree beside P(two heads) = 1/4
        stage.equation(r"P = \frac{1}{4}")
        stage.mark(min_seconds=6.05)
        self.wait(1)
