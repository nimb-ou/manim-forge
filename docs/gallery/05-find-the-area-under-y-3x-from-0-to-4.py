
        stage = Stage(self)
        # beat 1: Axes with y = 3x from 0 to 4
        stage.title("Area under y = 3x")
        ax = draw_axes(stage, x_range=(0, 4.5), y_range=(0, 15))
        f = lambda x: 3 * x
        g = plot_graph(stage, ax, f, label="y = 3x")
        stage.mark(min_seconds=5.05)

        # beat 2: The region under the line from 0 to 4 shaded: a triangle
        area = shade_area(stage, ax, g, 0, 4)
        h = draw_line(stage, ax.c2p(4, 0), ax.c2p(4, 12), color=YELLOW, dashed=True)
        stage.label(h, "12", direction=RIGHT, color=YELLOW)
        stage.caption("Base 4, height 12")
        stage.pause(1.5)
        stage.mark(min_seconds=5.17)

        # beat 3: Rectangles under the line close in on the triangle
        stage.clear(keep=[ax, g])
        rects = riemann_refine(stage, ax, g, 0, 4)
        stage.caption("The rectangles approach 24")
        stage.pause(1.5)
        stage.mark(min_seconds=6.30)

        # beat 4: Half base times height equals the integral
        stage.equation(r"\tfrac{1}{2} \cdot 4 \cdot 12 = 24", r"\int_0^4 3x \, dx = 24", where="right")
        stage.pause(1.5)
        stage.mark(min_seconds=6.85)
        self.wait(1)
