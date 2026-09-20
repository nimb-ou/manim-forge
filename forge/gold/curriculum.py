"""The gold-scene curriculum — what to build, in what order, and why.

Gold scenes are the only source of style in this project. Everything else
teaches the Manim API; these teach what makes an explanation good. So they are
planned as a curriculum rather than collected as a list, with three rules:

**Building blocks first.** A scene about eigenvectors is worthless as training
data if no scene has established what a linear transformation *looks like*.
Tier 1 is the visual grammar everything else assumes: number lines, functions
as mappings, area as a quantity you can rearrange.

**Breadth over depth.** Ten calculus scenes teach a model calculus notation.
Ten scenes across ten fields teach it how to explain. The tiers are balanced
across domains deliberately, even where one domain has more obvious material.

**Narration is not optional.** Every beat carries the line a narrator would
speak. It is what makes these explainers rather than diagrams, it is what the
planner model learns to produce, and it is the script a voice track will read
later — so it is written as speech, not as captions.

Status is tracked here so the curriculum and the repository cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scene:
    key: str
    title: str
    domain: str
    tier: int                 # 1 = visual grammar, 2 = core, 3 = composite
    prompt: str               # the one-line request a person would type
    needs: list[str] = field(default_factory=list)   # primitive modules
    builds_on: list[str] = field(default_factory=list)
    done: bool = False


CURRICULUM: list[Scene] = [

    # ── Tier 1 — visual grammar ──────────────────────────────────────────
    # Nothing here is hard. All of it is assumed by everything later, and a
    # model that has never seen a number line drawn well will draw one badly
    # inside a scene about something else.
    Scene("number_line", "The number line and what lives on it", "foundations", 1,
          "show a number line, place integers then fractions then an irrational "
          "on it, and show that between any two there is always another",
          needs=["numbers"], done=True),
    Scene("function_machine", "A function as a machine", "foundations", 1,
          "show a function as a machine taking an input and producing an output, "
          "then show the same function as a curve, and connect the two views",
          needs=["calculus"], done=True),
    Scene("coordinate_plane", "Two numbers locate a point", "foundations", 1,
          "build the coordinate plane from two number lines and show how a pair "
          "of numbers names exactly one point",
          needs=["plane"], done=True),
    Scene("area_rearranged", "Area survives rearrangement", "geometry", 1,
          "show that cutting a shape and rearranging the pieces leaves the area "
          "unchanged, and use that to turn a parallelogram into a rectangle",
          needs=["grid"], done=True),
    Scene("angle_basics", "Why a triangle's angles make a straight line", "geometry", 1,
          "show why the three angles of any triangle add to a straight line, by "
          "tearing off the corners and fitting them together", done=True),

    # ── Tier 2 — core ideas, one per field ───────────────────────────────
    Scene("pythagoras", "Pythagoras by rearrangement", "geometry", 2,
          "prove the Pythagorean theorem by rearranging four right triangles "
          "inside a square, twice, and comparing what is left",
          needs=["grid"], builds_on=["area_rearranged"],
          done=True),
    Scene("circle_area", "Unrolling a circle", "geometry", 2,
          "derive the area of a circle by slicing it into thin wedges and "
          "rearranging them into something that approaches a rectangle",
          builds_on=["area_rearranged"],
          done=True),
    Scene("linear_transform", "A matrix moves the whole plane", "vectors", 2,
          "show a linear transformation as the entire grid moving, with the "
          "basis vectors landing where the matrix columns say",
          needs=["vectors"], builds_on=["coordinate_plane"]),
    Scene("determinant", "The determinant is an area factor", "vectors", 2,
          "show that the determinant is the factor by which a transformation "
          "scales area, and what a negative one means",
          needs=["vectors"], builds_on=["linear_transform"]),
    Scene("eigenvectors", "The directions that do not turn", "vectors", 2,
          "show eigenvectors as the directions a transformation leaves pointing "
          "the same way, only stretched",
          needs=["vectors"], builds_on=["linear_transform"]),
    Scene("derivative_slope", "The derivative as a limit of slopes", "calculus", 2,
          "show the derivative as the slope of a line through two points on a "
          "curve, as those points slide together",
          needs=["calculus"], builds_on=["function_machine"],
          done=True),
    Scene("chain_rule", "Rates of change, nested", "calculus", 2,
          "show the chain rule as two gears of different sizes, where turning "
          "one turns the other at a multiplied rate",
          needs=["calculus"], builds_on=["derivative_slope"],
          done=True),
    Scene("taylor", "Approximating a curve with polynomials", "calculus", 2,
          "show a Taylor series approximating sine, adding one term at a time, "
          "and show the approximation improving then failing far from the centre",
          needs=["calculus"],
          done=True),
    Scene("euclid_gcd", "Euclid's algorithm as shrinking rectangles", "number-theory", 2,
          "show Euclid's algorithm for the greatest common divisor as repeatedly "
          "cutting the largest possible square from a rectangle",
          needs=["numbers"], builds_on=["area_rearranged"],
          done=True),
    Scene("modular_clock", "Arithmetic on a clock face", "number-theory", 2,
          "show modular arithmetic as counting around a clock face, and what "
          "happens when you keep adding the same number",
          needs=["numbers"],
          done=True),
    Scene("irrational_sqrt2", "Why root two is not a fraction", "number-theory", 2,
          "show geometrically why the square root of two cannot be written as a "
          "fraction, by infinite descent",
          needs=["numbers"],
          done=True),
    Scene("bayes", "Belief as area", "probability", 2,
          "show Bayes' theorem as areas in a rectangle, where evidence shrinks "
          "the space of possibilities",
          needs=["probability", "grid"],
          done=True),
    Scene("central_limit", "Why the bell curve keeps appearing", "probability", 2,
          "show many small random effects adding up into a bell curve, using a "
          "Galton board, and show it emerge regardless of the underlying shape",
          needs=["probability"], builds_on=["law_of_large_numbers"]),
    Scene("monty_hall", "The door problem, played out", "puzzles", 2,
          "play the Monty Hall problem out over many trials and show the "
          "switching strategy winning two thirds of the time",
          needs=["probability"],
          done=True),
    Scene("bfs_graph", "Searching outward one step at a time", "graphs", 2,
          "show breadth-first search expanding outward from a node in waves, "
          "and show that it finds the shortest path when every edge costs one",
          needs=["graphs"],
          done=True),
    Scene("dijkstra", "Shortest paths when edges cost different amounts", "graphs", 2,
          "show Dijkstra's algorithm relaxing edges and settling on the "
          "cheapest route, and why greedy works here",
          needs=["graphs"], builds_on=["bfs_graph"],
          done=True),
    Scene("binary_search", "Halving the haystack", "discrete", 2,
          "show binary search discarding half the remaining possibilities each "
          "step, and count how few steps that takes",
          done=True),
    Scene("recursion_hanoi", "A problem that contains itself", "discrete", 2,
          "solve the Towers of Hanoi recursively and show the move count "
          "doubling with each disc",
          done=True),
    Scene("projectile", "Two motions at once", "physics", 2,
          "show projectile motion as horizontal motion and vertical motion "
          "happening independently, and combine them into a parabola",
          needs=["oscillation"],
          done=True),
    Scene("field_lines", "What a field looks like", "physics", 2,
          "show the field around a charge as arrows, then as field lines, and "
          "show what happens when a second, opposite charge arrives",
          needs=["vectors"],
          done=True),
    Scene("wave_interference", "Two sources, one pattern", "physics", 2,
          "show two wave sources interfering, with bright lines where crests "
          "meet crests and dark lines where they cancel",
          needs=["oscillation"],
          done=True),
    Scene("orbit", "Falling forever around a planet", "physics", 2,
          "show an orbit as continuous falling, by firing a cannonball faster "
          "and faster until it misses the ground",
          needs=["oscillation"],
          done=True),
    Scene("electron_shells", "Why atoms fill up in layers", "chemistry", 2,
          "show electrons filling shells around a nucleus and why the shells "
          "hold two, eight and eight",
          done=True),
    Scene("covalent_bond", "Sharing electrons", "chemistry", 2,
          "show a covalent bond forming as two atoms share a pair of electrons, "
          "and why that lowers the energy",
          done=True),
    Scene("reaction_rate", "Concentration falling over time", "chemistry", 2,
          "show a reaction as particles colliding, and plot the concentration "
          "falling into an exponential decay",
          needs=["calculus"],
          done=True),
    Scene("gradient_descent_2d", "Downhill on a surface", "machine-learning", 2,
          "show gradient descent on a two-dimensional loss surface, following "
          "the steepest direction down into a valley",
          needs=["optimise"], builds_on=["learning_rate"],
          done=True),
    Scene("decision_boundary", "Drawing the line between two classes", "machine-learning", 2,
          "show a classifier learning a boundary between two clouds of points, "
          "and what happens when the clouds overlap",
          needs=["vectors"],
          done=True),
    Scene("neuron", "One neuron, one decision", "machine-learning", 2,
          "show a single neuron taking a weighted sum and squashing it, and how "
          "changing the weights tilts the boundary it draws",
          needs=["vectors"], builds_on=["decision_boundary"],
          done=True),
    Scene("convolution", "A kernel sliding over an image", "machine-learning", 2,
          "show a convolution kernel sliding across a grid of pixels and what "
          "different kernels detect",
          needs=["grid"],
          done=True),
    Scene("platonic_solids", "The only five", "shapes-3d", 2,
          "show the five Platonic solids and why there cannot be a sixth",
          needs=["solids"]),
    Scene("conic_sections", "One cone, four curves", "shapes-3d", 2,
          "slice a cone at different angles to produce a circle, an ellipse, a "
          "parabola and a hyperbola",
          needs=["solids"]),
    Scene("projection_3d", "A shadow is a projection", "shapes-3d", 2,
          "show a three-dimensional object casting a two-dimensional shadow, "
          "and how rotating it changes what the shadow reveals",
          needs=["solids"]),

    # ── Tier 3 — composite, where ideas meet ─────────────────────────────
    Scene("fourier", "Building any wave from circles", "calculus", 3,
          "build a square wave out of rotating circles, adding one frequency at "
          "a time",
          needs=["oscillation", "calculus"], builds_on=["taylor", "wave_interference"]),
    Scene("fundamental_theorem", "Area and slope are inverse", "calculus", 3,
          "show that accumulating area under a curve and taking a slope undo "
          "each other",
          needs=["calculus"], builds_on=["derivative_slope", "riemann"]),
    Scene("change_of_basis", "The same point, two descriptions", "vectors", 3,
          "show one vector described in two different coordinate systems and "
          "how to translate between them",
          needs=["vectors"], builds_on=["linear_transform", "eigenvectors"]),
    Scene("matrix_composition", "Matrix multiplication is doing one then another", "vectors", 3,
          "show matrix multiplication as applying one transformation then a "
          "second, and why the order matters",
          needs=["vectors"], builds_on=["linear_transform"]),
    Scene("pca", "Finding the direction that matters", "machine-learning", 3,
          "show principal component analysis finding the direction of greatest "
          "spread in a cloud of points",
          needs=["vectors"], builds_on=["eigenvectors", "decision_boundary"]),
    Scene("backprop", "Error flowing backwards", "machine-learning", 3,
          "show error propagating backwards through a small network, adjusting "
          "each weight by how much it contributed",
          needs=["optimise"], builds_on=["neuron", "chain_rule"]),
    Scene("divergence_curl", "Sources, sinks and swirls", "physics", 3,
          "show divergence as flow out of a small region and curl as rotation, "
          "in the same vector field",
          needs=["vectors"], builds_on=["field_lines"]),
    Scene("entropy", "Counting the ways", "physics", 3,
          "show entropy as the number of arrangements that look the same, using "
          "particles in a box",
          needs=["probability", "grid"]),
    Scene("collatz", "The sequence nobody can explain", "number-theory", 3,
          "show the Collatz sequence bouncing for many starting numbers and the "
          "tree they all fall into",
          needs=["numbers"]),
    Scene("golden_ratio", "The spiral hiding in Fibonacci", "number-theory", 3,
          "show the Fibonacci sequence, the ratio of consecutive terms settling, "
          "and the spiral it draws",
          needs=["numbers", "grid"]),
    Scene("graph_colouring", "The fewest colours that work", "graphs", 3,
          "colour a graph so no two connected nodes match, using as few colours "
          "as possible, and show why it is hard",
          needs=["graphs"], builds_on=["bfs_graph"],
          done=True),
    Scene("konigsberg", "The walk that cannot be done", "graphs", 3,
          "show the seven bridges of Konigsberg and prove no route crosses each "
          "exactly once, by counting odd degrees",
          needs=["graphs"],
          done=True),
    Scene("pigeonhole", "More items than containers", "puzzles", 3,
          "show the pigeonhole principle and use it to prove something "
          "surprising about a group of people",
          needs=["grid", "probability"]),
    Scene("birthday_paradox", "Why twenty-three is enough", "puzzles", 3,
          "show why a room of twenty-three people probably contains a shared "
          "birthday, by counting pairs rather than people",
          needs=["probability"]),
    Scene("sorting", "Comparing and swapping", "discrete", 3,
          "show two sorting algorithms racing on the same data and why one "
          "scales better",
          needs=["grid"], builds_on=["binary_search"]),
    Scene("combinatorics", "Choosing versus arranging", "discrete", 3,
          "show the difference between permutations and combinations by laying "
          "out every possibility for a small set",
          needs=["grid", "probability"]),

    # ── already built ────────────────────────────────────────────────────
    Scene("rook_corners", "Rook mobility and corner probability", "discrete", 2,
          "built", needs=["grid"], done=True),
    Scene("learning_rate", "Learning rate in gradient descent", "machine-learning", 2,
          "built", needs=["optimise"], done=True),
    Scene("dot_product", "The dot product as projection", "vectors", 2,
          "built", needs=["vectors"], done=True),
    Scene("sphere_cube", "The largest cube inside a sphere", "shapes-3d", 2,
          "built", needs=["solids"], done=True),
    Scene("riemann", "Riemann sums converging", "calculus", 2,
          "built", needs=["calculus"], done=True),
    Scene("spring", "A mass on a spring", "physics", 2,
          "built", needs=["oscillation"], done=True),
    Scene("sieve", "The sieve of Eratosthenes", "number-theory", 2,
          "built", needs=["numbers"], done=True),
    Scene("law_of_large_numbers", "The law of large numbers", "probability", 2,
          "built", needs=["probability"], done=True),
]


def summary() -> dict:
    from collections import Counter
    done = [s for s in CURRICULUM if s.done]
    return {
        "total": len(CURRICULUM),
        "done": len(done),
        "remaining": len(CURRICULUM) - len(done),
        "by_tier": dict(sorted(Counter(s.tier for s in CURRICULUM).items())),
        "by_domain": dict(Counter(s.domain for s in CURRICULUM).most_common()),
        "done_by_domain": dict(Counter(s.domain for s in done).most_common()),
    }


def next_up(n: int = 8) -> list[Scene]:
    """What to build next: lowest tier first, and only once its
    prerequisites are done — a scene about eigenvectors is worthless as
    training data if nothing has shown what a transformation looks like."""
    done_keys = {s.key for s in CURRICULUM if s.done}
    ready = [s for s in CURRICULUM
             if not s.done and all(b in done_keys for b in s.builds_on)]
    ready.sort(key=lambda s: (s.tier, s.domain))
    return ready[:n]


if __name__ == "__main__":
    import json
    s = summary()
    print(json.dumps(s, indent=2))
    print("\nnext up:")
    for sc in next_up(10):
        print(f"  [T{sc.tier}] {sc.domain:<17} {sc.title}")
