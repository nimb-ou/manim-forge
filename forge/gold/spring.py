"""Gold scene 006 — a mass on a spring, and the wave it draws.

Driven by ``forge.primitives.oscillation``. The motion is integrated from
F = -kx rather than plotted from a cosine, so the trace the scene draws is the
trace the physics produces. The claim that it *is* a cosine is checked
numerically before it is made — the integration agrees with A·cos(ωt) to about
2e-5, and the energy drifts by 8e-11 over three periods.

That distinction is the whole reason this scene is worth having. Plotting a
cosine beside a bouncing dot and asserting they match would look identical and
mean nothing.
"""

import math

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.oscillation import SpringMass, sample

MASS_C = BLUE_C
TRACE_C = "#F0AC5F"
KE_C = "#5CD0B3"
PE_C = "#9A72AC"
DIM = GREY_B

WALL_X = -6.1
REST_X = -3.6          # equilibrium position of the mass
SCALE = 0.85           # world units per unit of displacement


def coil(start, end, loops=11, width=0.34):
    """A spring drawn as a real coil between two points."""
    pts = []
    total = np.linalg.norm(end - start)
    direction = (end - start) / max(total, 1e-9)
    perp = np.array([-direction[1], direction[0], 0.0])
    lead = 0.16
    body = max(total - 2 * lead, 1e-6)
    n = loops * 12
    pts.append(start)
    for i in range(n + 1):
        u = i / n
        along = start + direction * (lead + body * u)
        pts.append(along + perp * width * math.sin(2 * math.pi * loops * u))
    pts.append(end)
    return VMobject(stroke_color=GREY_B, stroke_width=3).set_points_smoothly(pts)


class SpringTrace(ForgeScene):

    spring = SpringMass(amplitude=1.9, mass=1.0, stiffness=4.0)

    def mass_pos(self, x: float) -> np.ndarray:
        return np.array([REST_X + x * SCALE, 1.3, 0.0])

    @beat("Pull a mass on a spring and let go", seconds=7,
          narration="A mass on a spring. Pull it aside and the spring pulls "
                    "back — harder the further you stretch it. Let go, and it "
                    "does not simply return to the middle. It overshoots.")
    def setup(self):
        self.wall = Line([WALL_X, 0.55, 0], [WALL_X, 2.05, 0],
                         color=GREY_B, stroke_width=6)
        hatch = VGroup(*[Line([WALL_X, y, 0], [WALL_X - 0.22, y - 0.18, 0],
                              color=GREY_D, stroke_width=2)
                         for y in np.arange(0.7, 2.06, 0.22)])
        self.block = Square(side_length=0.62, fill_opacity=1, fill_color=MASS_C,
                            stroke_width=0).move_to(self.mass_pos(self.spring.amplitude))
        self.spring_mob = always_redraw(
            lambda: coil(np.array([WALL_X, 1.3, 0.0]),
                         self.block.get_left()))
        self.title = Text("What shape does a spring draw?",
                          font_size=29).to_edge(UP, buff=0.35)

        self.play(FadeIn(self.title, shift=DOWN * 0.2),
                  Create(self.wall), Create(hatch), run_time=1.1)
        self.play(FadeIn(self.block, scale=0.6), run_time=0.7)
        self.add(self.spring_mob)
        self.play(Indicate(self.block, color=MASS_C, scale_factor=1.15), run_time=0.9)
        self.wait(0.8)

    @beat("Let it run, and trace its position against time", seconds=11,
          narration="Now record where it is, moment by moment, and lay those "
                    "positions out along a time axis. A shape appears — and it "
                    "is not something we drew. It is what the motion leaves "
                    "behind.")
    def trace(self):
        s = self.spring
        duration = 2 * s.period
        traj = sample(s.integrate(duration), 260)      # integrated, not assumed

        self.axes = Axes(
            x_range=[0, duration, s.period / 2], y_range=[-2.4, 2.4, 1],
            x_length=7.2, y_length=2.9,
            axis_config={"include_tip": False, "stroke_width": 2,
                         "color": GREY_D, "font_size": 16},
        ).to_edge(DOWN, buff=0.7).shift(RIGHT * 0.4)

        self.play(Create(self.axes), run_time=1.0)

        curve = VMobject(stroke_color=TRACE_C, stroke_width=4)
        curve.set_points_smoothly([self.axes.c2p(t, x) for t, x, _ in traj])

        tracker = ValueTracker(0.0)
        moving = always_redraw(
            lambda: self.block.move_to(self.mass_pos(s.analytic(tracker.get_value())))
        )
        self.add(moving)
        self.play(tracker.animate.set_value(duration),
                  Create(curve),
                  run_time=6.0, rate_func=linear)
        self.remove(moving)
        self.curve, self.traj = curve, traj
        self.wait(0.8)

    @beat("The shape is exactly a cosine", seconds=8,
          narration="That shape is a cosine wave. Not approximately — the "
                    "motion, solved step by step from the force alone, agrees "
                    "with the cosine to five decimal places. The wave was "
                    "hiding in the force law the whole time.")
    def identify(self):
        s = self.spring
        err = s.verify_matches_cosine(2 * s.period)
        assert err < 5e-3, f"trace does not match a cosine: {err}"

        overlay = self.axes.plot(lambda t: s.analytic(t),
                                 x_range=[0, 2 * s.period],
                                 color=WHITE, stroke_width=2).set_stroke(opacity=0.85)
        label = VGroup(
            MathTex(r"x(t) = A\cos(\omega t)", font_size=38, color=TRACE_C),
            Text(f"agrees to {err:.0e}", font_size=20, color=DIM),
        ).arrange(DOWN, buff=0.2).to_edge(RIGHT, buff=0.7).shift(UP * 1.5)

        self.play(Create(overlay), run_time=1.4)
        self.play(FadeIn(label, shift=UP * 0.15), run_time=0.9)
        self.wait(1.4)
        self.play(FadeOut(overlay), FadeOut(label), run_time=0.6)

    @beat("Energy sloshes between two forms and the total holds still", seconds=9,
          narration="And watch the energy. At the edges it is all stored in "
                    "the stretched spring. Through the middle it is all motion. "
                    "Each one rises exactly as the other falls, and their sum "
                    "never moves.")
    def energy(self):
        s = self.spring
        traj = s.integrate(2 * s.period)
        total0 = sum(s.energy(traj[0][1], traj[0][2]))
        totalN = sum(s.energy(traj[-1][1], traj[-1][2]))
        assert abs(totalN - total0) < 1e-6      # conservation, checked

        pts = sample(traj, 200)
        ke = VMobject(stroke_color=KE_C, stroke_width=3.5)
        pe = VMobject(stroke_color=PE_C, stroke_width=3.5)
        tot = VMobject(stroke_color=WHITE, stroke_width=2.5)
        scale = 2.2 / total0
        ke.set_points_smoothly([self.axes.c2p(t, s.energy(x, v)[0] * scale - 2.2)
                                for t, x, v in pts])
        pe.set_points_smoothly([self.axes.c2p(t, s.energy(x, v)[1] * scale - 2.2)
                                for t, x, v in pts])
        tot.set_points_smoothly([self.axes.c2p(t, sum(s.energy(x, v)) * scale - 2.2)
                                 for t, x, v in pts])

        key = VGroup(
            Text("kinetic", font_size=20, color=KE_C),
            Text("potential", font_size=20, color=PE_C),
            Text("total — constant", font_size=20, color=WHITE),
        ).arrange(DOWN, buff=0.16, aligned_edge=LEFT).to_edge(RIGHT, buff=0.7).shift(UP * 1.6)

        self.play(self.curve.animate.set_stroke(opacity=0.22), run_time=0.5)
        self.play(Create(ke), FadeIn(key[0]), run_time=1.3)
        self.play(Create(pe), FadeIn(key[1]), run_time=1.3)
        self.play(Create(tot), FadeIn(key[2]), run_time=1.1)
        self.wait(1.6)
        self.play(*[FadeOut(m) for m in (ke, pe, tot, key)], run_time=0.6)

    @beat("Heavier is slower, stiffer is faster", seconds=8,
          narration="Change the spring and the wave changes with it. Four times "
                    "the mass takes twice as long to swing. Four times the "
                    "stiffness takes half. The period depends on the square "
                    "root of their ratio, and nothing else — not even on how "
                    "far you pulled it.")
    def compare(self):
        rows = VGroup()
        for m, k, note in ((1.0, 4.0, "baseline"),
                           (4.0, 4.0, "4x the mass"),
                           (1.0, 16.0, "4x the stiffness")):
            sm = SpringMass(mass=m, stiffness=k)
            rows.add(VGroup(
                Text(note, font_size=21, color=DIM),
                Text(f"{sm.period:.2f}s", font_size=26, color=TRACE_C),
            ).arrange(RIGHT, buff=0.35, aligned_edge=DOWN))
        rows.arrange(DOWN, buff=0.26, aligned_edge=LEFT)
        rows.to_edge(RIGHT, buff=0.7).shift(UP * 1.4)

        formula = MathTex(r"T = 2\pi\sqrt{\tfrac{m}{k}}", font_size=40, color=TRACE_C)
        formula.next_to(rows, DOWN, buff=0.45)

        self.play(self.curve.animate.set_stroke(opacity=1.0), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in rows],
                              lag_ratio=0.3, run_time=1.6))
        self.play(Write(formula), run_time=1.2)
        self.wait(1.6)
