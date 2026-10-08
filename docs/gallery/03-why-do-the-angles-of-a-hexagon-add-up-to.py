
        stage = Stage(self)
        # beat 1: A regular hexagon
        stage.title("Sum of hexagon angles")
        pts = [(2 * np.cos(TAU * k / 6), 2 * np.sin(TAU * k / 6)) for k in range(6)]
        hexa = draw_polygon(stage, pts, where=None)
        stage.mark(min_seconds=6.57)

        # beat 2: Lines from one corner splitting the hexagon into 4 triangles
        d1 = draw_line(stage, pts[0], pts[2], dashed=True)
        d2 = draw_line(stage, pts[0], pts[3], dashed=True)
        d3 = draw_line(stage, pts[0], pts[4], dashed=True)
        stage.caption("4 triangles")
        stage.mark(min_seconds=8.80)

        # beat 3: The equation 4 × 180° = 720°, 720° ÷ 6 = 120° beside the hexagon
        stage.equation(r"4 \times 180^\circ = 720^\circ", r"720^\circ \div 6 = 120^\circ")
        stage.mark(min_seconds=13.25)
        self.wait(1)
