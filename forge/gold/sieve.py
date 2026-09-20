"""Gold scene 007 — the sieve of Eratosthenes.

Driven by ``forge.primitives.numbers``. The scene replays the strikes the
algorithm actually made, in the order it made them, and the surviving primes
are cross-checked against trial division before the scene names them. A scene
that showed only the final primes would have skipped the algorithm, which is
the part worth watching.
"""

from manim import *

from forge.beats import ForgeScene, beat
from forge.primitives.numbers import sieve, verify_sieve

LIMIT = 100
COLS = 10
CELL = 0.52
GRID_X = -3.0
PANEL_X = 3.6

ALIVE_C = "#2A313A"
PRIME_C = "#5CD0B3"
STRIKE_C = {2: "#FC6255", 3: "#F0AC5F", 5: "#9A72AC", 7: "#58C4DD"}
DIM = GREY_B


class Sieve(ForgeScene):

    primes, strikes = sieve(LIMIT)

    def cell_pos(self, n: int) -> np.ndarray:
        i = n - 1
        r, c = divmod(i, COLS)
        return np.array([GRID_X + (c - (COLS - 1) / 2) * CELL,
                         2.4 - r * CELL, 0.0])

    @beat("Lay out every number up to a hundred", seconds=6,
          narration="Every whole number from one to a hundred. Somewhere in "
                    "here are the primes — the numbers with no factors but "
                    "themselves and one. Rather than testing each one, we can "
                    "remove everything that cannot possibly be prime.")
    def grid(self):
        self.tiles, self.labels = {}, {}
        for n in range(1, LIMIT + 1):
            sq = Square(side_length=CELL * 0.92, stroke_width=0,
                        fill_color=ALIVE_C, fill_opacity=1).move_to(self.cell_pos(n))
            tx = Text(str(n), font_size=15, color=GREY_B).move_to(self.cell_pos(n))
            self.tiles[n], self.labels[n] = sq, tx
        self.title = Text("Sieving out the primes", font_size=29).to_edge(UP, buff=0.3)

        self.play(FadeIn(self.title, shift=DOWN * 0.2), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(self.tiles[n]) for n in range(1, LIMIT + 1)],
                              lag_ratio=0.004, run_time=1.8))
        self.play(LaggedStart(*[FadeIn(self.labels[n]) for n in range(1, LIMIT + 1)],
                              lag_ratio=0.002, run_time=1.2))
        # 1 is neither prime nor composite; removing it first avoids the
        # classic off-by-one the sieve is famous for.
        self.play(self.tiles[1].animate.set_fill(opacity=0.12),
                  self.labels[1].animate.set_opacity(0.25), run_time=0.6)
        self.wait(0.5)

    @beat("Strike every multiple of two, then three, five and seven", seconds=16,
          narration="Take the first survivor, two. Every multiple of two after "
                    "it is composite, so cross them all out. Then the next "
                    "survivor, three. Then five, then seven. Notice we start "
                    "each pass at the prime squared — anything smaller was "
                    "already caught by a smaller prime.")
    def strike(self):
        by_prime: dict[int, list[int]] = {}
        for s in self.strikes:
            by_prime.setdefault(s.prime, []).append(s.value)

        for p in sorted(by_prime):
            colour = STRIKE_C.get(p, GREY_D)
            head = VGroup(
                Text(str(p), font_size=52, color=colour),
                Text(f"striking from {p*p}", font_size=19, color=DIM),
            ).arrange(DOWN, buff=0.16).move_to([PANEL_X, 1.4, 0])
            self.play(FadeIn(head, scale=0.85), run_time=0.5)
            self.play(Indicate(self.tiles[p], color=colour, scale_factor=1.3),
                      run_time=0.4)

            hits = by_prime[p]
            self.play(LaggedStart(*[
                AnimationGroup(
                    self.tiles[v].animate.set_fill(colour, opacity=0.30),
                    self.labels[v].animate.set_opacity(0.22),
                ) for v in hits],
                lag_ratio=min(0.06, 1.2 / max(len(hits), 1)),
                run_time=min(2.4, 0.35 + 0.05 * len(hits))))
            self.play(FadeOut(head), run_time=0.3)
        self.wait(0.6)

    @beat("What survives is exactly the primes", seconds=9,
          narration="Four passes, and everything composite is gone. What is "
                    "left are the twenty-five primes below a hundred — not "
                    "tested one by one, but whatever the sieve failed to "
                    "remove.")
    def survivors(self):
        assert verify_sieve(LIMIT)          # cross-checked before claiming

        self.play(LaggedStart(*[
            AnimationGroup(self.tiles[p].animate.set_fill(PRIME_C, opacity=0.9),
                           self.labels[p].animate.set_color(BLACK).set_opacity(1.0))
            for p in self.primes],
            lag_ratio=0.03, run_time=2.4))

        count = VGroup(
            Text(str(len(self.primes)), font_size=74, color=PRIME_C),
            Text("primes below 100", font_size=22, color=DIM),
        ).arrange(DOWN, buff=0.18).move_to([PANEL_X, 0.9, 0])
        self.play(FadeIn(count[0], scale=0.7), Write(count[1]), run_time=1.0)
        self.wait(1.6)
