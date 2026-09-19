"""Gold scene 004 — the largest cube inside a sphere.

From Nimit's prompt: a sphere built from dots, a cube fitted inside it, and
the leftover volume calculated.

Driven by ``forge.primitives.solids``. The cube's edge is 2R/sqrt(3) because
its *space* diagonal is the sphere's diameter — and the primitive asserts all
eight vertices lie on the surface before the scene claims they do. A cube drawn
slightly too large still looks like a cube inside a sphere; nothing on screen
would give it away, which is exactly why it is checked.

Dots are placed on a Fibonacci spiral rather than a latitude/longitude grid.
Uniform sampling in those angles crowds the poles and thins the equator, which
reads as a mistake even to a viewer who could not say why.

Note on 3D: text must be registered with ``add_fixed_in_frame_mobjects`` or the
camera rotation drags it off with the scene.
"""

import math

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.solids import SphereCube, fibonacci_sphere

DOT_C = BLUE_C
CUBE_C = "#F0AC5F"
GAP_C = "#5CD0B3"
DIM = GREY_B

SCALE = 2.2          # world units per unit radius
N_DOTS = 320


class SphereInCube(ForgeScene, ThreeDScene):

    solid = SphereCube(radius=1.0)

    def p3(self, x, y, z):
        return np.array([x, y, z]) * SCALE

    def label(self, mob):
        """Register a 2D overlay so camera rotation leaves it alone."""
        self.add_fixed_in_frame_mobjects(mob)
        return mob

    @beat("Build a sphere out of points", seconds=9,
          narration="Start with a sphere — not a solid surface, but a few "
                    "hundred points scattered evenly across it. Every one of "
                    "them sits at exactly the same distance from the centre. "
                    "That is all a sphere is.")
    def build_sphere(self):
        self.set_camera_orientation(phi=68 * DEGREES, theta=-55 * DEGREES, zoom=0.95)

        pts = fibonacci_sphere(N_DOTS, self.solid.radius)
        self.dots = VGroup(*[
            Dot3D(self.p3(*p), radius=0.028, color=DOT_C).set_opacity(0.85)
            for p in pts])

        self.title = self.label(
            Text("The largest cube that fits inside a sphere", font_size=26)
            .to_edge(UP, buff=0.4))

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.8)
        # Grown from the centre outward: the dots arrive as a sphere forming,
        # rather than simply appearing.
        self.play(LaggedStart(*[GrowFromPoint(d, ORIGIN) for d in self.dots],
                              lag_ratio=0.004, run_time=3.4))
        self.begin_ambient_camera_rotation(rate=0.10)
        self.wait(1.6)

    @beat("Fit the biggest possible cube inside it", seconds=8,
          narration="Now slide the biggest cube you can inside. It cannot grow "
                    "any further, because its eight corners have reached the "
                    "surface. Those corners are what limit it.")
    def add_cube(self):
        assert self.solid.verify()          # vertices genuinely on the sphere

        s = self.solid.side * SCALE
        self.cube = Cube(side_length=s, fill_opacity=0.18, fill_color=CUBE_C,
                         stroke_color=CUBE_C, stroke_width=2.5)
        self.corner_dots = VGroup(*[
            Dot3D(self.p3(*v), radius=0.055, color=CUBE_C)
            for v in self.solid.vertices()])

        self.play(FadeIn(self.cube, scale=0.4), run_time=1.8)
        self.play(LaggedStart(*[Flash(d, color=CUBE_C, line_length=0.12,
                                      flash_radius=0.16, num_lines=8)
                                for d in self.corner_dots],
                              lag_ratio=0.1, run_time=1.6))
        self.play(FadeIn(self.corner_dots), run_time=0.6)
        self.wait(1.4)

    @beat("The space diagonal is the diameter", seconds=8,
          narration="Here is the key. Draw the line from one corner of the cube "
                    "to the corner furthest away. It passes through the centre, "
                    "and both ends are on the sphere — so that diagonal is the "
                    "diameter. A cube's space diagonal is its edge times root "
                    "three, which fixes the edge at two R over root three.")
    def diagonal(self):
        v = self.solid.vertices()
        far = self.p3(*v[0]), self.p3(*v[-1])       # opposite corners
        self.diag = Line3D(*far, color=GAP_C, thickness=0.022)

        self.play(Create(self.diag), run_time=1.6)

        formula = self.label(
            MathTex(r"s\sqrt{3} = 2R", r"\;\Rightarrow\;", r"s = \frac{2R}{\sqrt{3}}",
                    font_size=38).to_edge(DOWN, buff=0.6))
        formula[2].set_color(CUBE_C)
        self.play(Write(formula), run_time=1.8)
        self.formula = formula
        self.wait(2.0)

    @beat("Compare the two volumes", seconds=10,
          narration="Now the volumes. The sphere is four thirds pi R cubed — "
                    "about four point one nine. The cube is its edge cubed — "
                    "about one point five four. So the cube, for all that it "
                    "fills the middle, takes up only thirty-seven percent of "
                    "the sphere.")
    def volumes(self):
        sc = self.solid
        self.play(FadeOut(self.formula), FadeOut(self.diag), run_time=0.7)
        self.stop_ambient_camera_rotation()
        self.move_camera(phi=64 * DEGREES, theta=-30 * DEGREES, run_time=1.6)

        rows = VGroup(
            VGroup(Text("sphere", font_size=24, color=DOT_C),
                   MathTex(rf"{sc.v_sphere:.2f}", font_size=34, color=DOT_C)),
            VGroup(Text("cube", font_size=24, color=CUBE_C),
                   MathTex(rf"{sc.v_cube:.2f}", font_size=34, color=CUBE_C)),
        )
        for r in rows:
            r.arrange(RIGHT, buff=0.45, aligned_edge=DOWN)
        rows.arrange(DOWN, buff=0.3, aligned_edge=LEFT).to_corner(UR, buff=0.7)
        self.label(rows)

        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.25) for r in rows],
                              lag_ratio=0.35, run_time=1.8))

        frac = self.label(
            Text(f"the cube fills {sc.cube_fraction:.0%}", font_size=26, color=CUBE_C)
            .next_to(rows, DOWN, buff=0.45).align_to(rows, RIGHT))
        self.play(FadeIn(frac, shift=UP * 0.15), run_time=0.9)
        self.rows, self.frac = rows, frac
        self.wait(1.6)

    @beat("What is left over", seconds=9,
          narration="Which leaves two point six five cubic units of empty space "
                    "between the flat faces of the cube and the curve of the "
                    "sphere. Nearly two thirds of the sphere is in that gap — "
                    "far more than the cube that looked like it filled it.")
    def leftover(self):
        sc = self.solid
        self.play(self.dots.animate.set_opacity(0.3),
                  self.cube.animate.set_fill(opacity=0.45), run_time=1.2)

        gap = self.label(
            VGroup(
                MathTex(rf"{sc.v_sphere:.2f} - {sc.v_cube:.2f} = {sc.v_gap:.2f}",
                        font_size=40, color=GAP_C),
                Text("left over between cube and sphere", font_size=22, color=DIM),
                Text(f"{1 - sc.cube_fraction:.0%} of the sphere", font_size=26, color=GAP_C),
            ).arrange(DOWN, buff=0.25).to_edge(DOWN, buff=0.55))

        self.play(FadeOut(self.rows), FadeOut(self.frac), run_time=0.5)
        self.play(Write(gap[0]), run_time=1.4)
        self.play(FadeIn(gap[1]), run_time=0.6)
        self.play(FadeIn(gap[2], shift=UP * 0.12), run_time=0.8)
        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(2.4)
        self.stop_ambient_camera_rotation()
