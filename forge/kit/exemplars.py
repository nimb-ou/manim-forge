"""Worked kit scenes, written by hand: training rows and in-prompt examples.

scripts/claude_kit_scenes.py renders them into training rows; with
Options(exemplar=True) the pipeline shows the coder the scene closest to
the request as a worked example. Topics are deliberately *not* those of
forge/evaluate/heldout_prompts.json.
"""
from __future__ import annotations

import re

# (request, [(intent, narration, code), ...])
SCENES: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("prove the Pythagorean theorem by putting a square on each side of a right triangle", [
        ("A right triangle with legs 3 and 4",
         "Start with a right triangle: one leg 3 long, the other 4.",
         'stage.title("A right triangle")\n'
         'tri = draw_right_triangle(stage, 3, 4)\n'
         'stage.caption("Legs 3 and 4, hypotenuse c")'),
        ("Build a square on each of the three sides",
         "Now build a square on every side, each as wide as the side it sits on.",
         'stage.clear()\n'
         'sq = squares_on_sides(stage, 3, 4)\n'
         'stage.caption("A square on every side")'),
        ("The two small squares add up to the big one",
         "Nine and sixteen make twenty-five: the two small squares exactly fill the big one.",
         'highlight(stage, sq)\n'
         'stage.equation("9 + 16 = 25", "a^2 + b^2 = c^2", where="right")'),
    ]),
    ("why is a triangle with sides 5, 12 and 13 a right triangle", [
        ("Squares on the sides of a 5-12-13 triangle",
         "Put a square on each side of the triangle with legs 5 and 12.",
         'stage.title("5, 12, 13")\n'
         'sq = squares_on_sides(stage, 5, 12, where="left")\n'
         'stage.caption("Squares on all three sides")'),
        ("Their areas satisfy a squared plus b squared equals c squared",
         "25 plus 144 is 169, which is 13 squared, so the angle between the legs is right.",
         'stage.equation("25 + 144 = 169", "5^2 + 12^2 = 13^2", where="right")\n'
         'pulse(stage, sq)'),
    ]),
    ("show that the three angles of any triangle add up to a straight line", [
        ("A triangle with its three angles coloured",
         "Take any triangle and colour its three corners.",
         'stage.title("Three angles")\n'
         'pic = angle_sum(stage)\n'
         'stage.caption("Laid side by side, they make a straight line")'),
        ("A different triangle gives the same straight line",
         "Change the triangle completely: the three angles still fill a half turn.",
         'stage.clear()\n'
         'pic2 = angle_sum(stage, [(-3, -1), (2, -1), (-2, 1.8)])\n'
         'stage.caption("Any triangle: 180 degrees")'),
        ("Write the rule",
         "So alpha plus beta plus gamma is always 180 degrees.",
         'stage.equation(r"\\alpha + \\beta + \\gamma = 180^\\circ", where="right")'),
    ]),
    ("what is the sum of the interior angles of a triangle, shown visually", [
        ("Lift the angles off the corners onto a line",
         "Cut the three corners off a triangle and slide them together on a line.",
         'stage.title("Interior angles")\n'
         'pic = angle_sum(stage, [(-2.5, -1.4), (3, -1.4), (-0.5, 1.9)])\n'
         'stage.caption("The corners meet in a half turn")'),
    ]),
    ("how do you count to fifteen with four bits", [
        ("Four bits with place values 8, 4, 2, 1 counting up",
         "Each bit is worth twice the one to its right: 1, 2, 4, 8. Count up and watch them flip.",
         'stage.title("Counting in binary")\n'
         'bits = count_binary(stage, 4, upto=15)\n'
         'stage.caption("All four bits on: 8 + 4 + 2 + 1 = 15")'),
        ("Read one number off the bits",
         "1101 means one eight, one four, no two and one one: thirteen.",
         'stage.equation("1101_2", "8 + 4 + 0 + 1", "13", where="right")'),
    ]),
    ("what does the binary number 101 mean", [
        ("Three bits counting from zero to five",
         "With three bits the places are worth 4, 2 and 1. Count up to five.",
         'stage.title("Binary 101")\n'
         'bits = count_binary(stage, 3, upto=5)'),
        ("101 is four plus one",
         "The pattern 101 has the four and the one switched on: five.",
         'stage.equation("101_2 = 4 + 0 + 1", "= 5", where="right")\n'
         'stage.caption("Each place is a power of two")'),
    ]),
    ("why can one byte hold 256 different values", [
        ("Eight bits counting up",
         "A byte is eight bits, each doubling the last: 1, 2, 4, up to 128.",
         'stage.title("A byte")\n'
         'bits = count_binary(stage, 8, upto=10)'),
        ("Each bit doubles the number of patterns",
         "Every extra bit doubles the count, so eight bits give two to the eighth patterns.",
         'stage.equation("2 \\times 2 \\times \\cdots \\times 2", "2^8 = 256", where="right")\n'
         'stage.caption("0 to 255: 256 values")'),
    ]),
    ("the derivative as the limit of slopes of secant lines", [
        ("A curve on axes",
         "Here is the curve y equals x squared over two.",
         'stage.title("Slope at a point")\n'
         'ax = draw_axes(stage, x_range=(-1, 4), y_range=(-1, 9))\n'
         'f = lambda x: x ** 2 / 2\n'
         'g = plot_graph(stage, ax, f, label="y = x^2/2")'),
        ("A secant through two points shrinks to the tangent",
         "Draw a line through two points on the curve, then slide the second point toward the first.",
         'sec = secant_to_tangent(stage, ax, f, 1)\n'
         'stage.caption("As h shrinks, the secant becomes the tangent")'),
        ("The tangent slides along the curve",
         "Do this at every point: the slope of the tangent changes as we move.",
         'stage.clear(keep=[ax, g])\n'
         'tan = slide_tangent(stage, ax, f, 0, 3)'),
        ("The limit definition",
         "That limit of secant slopes is the derivative.",
         'stage.equation(r"f\'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}", where="right")'),
    ]),
    ("find the slope of sin x at zero by shrinking a secant", [
        ("The sine curve",
         "Plot sine of x near zero.",
         'stage.title("Slope of sin x at 0")\n'
         'ax = draw_axes(stage, x_range=(-3, 3), y_range=(-1.5, 1.5))\n'
         'g = plot_graph(stage, ax, np.sin)'),
        ("A secant from zero closes in on the tangent",
         "Join zero to a point h away and let h go to zero; the slope settles at one.",
         'sec = secant_to_tangent(stage, ax, np.sin, 0, h=1.5)\n'
         'stage.equation(r"\\frac{\\sin h}{h} \\to 1", where="right")'),
    ]),
    ("why does a swinging pendulum trace out a cosine wave", [
        ("A pendulum swinging with its angle plotted against time",
         "Let a pendulum swing and record its angle as time goes on.",
         'stage.title("A swinging pendulum")\n'
         'ax, trace = swing_pendulum(stage, amplitude=0.5, swings=2)\n'
         'stage.caption("The angle rises and falls like a cosine")'),
        ("The formula for small swings",
         "For small swings the angle is a cosine of time.",
         'stage.equation(r"\\theta(t) = \\theta_0 \\cos(\\omega t)", where="right")'),
    ]),
    ("simple harmonic motion: how a longer period changes a pendulum", [
        ("A pendulum with a two-second period",
         "First a pendulum that takes two seconds per swing.",
         'stage.title("Period")\n'
         'ax, trace = swing_pendulum(stage, amplitude=0.4, swings=1.5, period=2)'),
        ("The same pendulum with a three-second period",
         "A longer string swings more slowly: the wave on the right stretches out.",
         'stage.clear()\n'
         'ax2, trace2 = swing_pendulum(stage, amplitude=0.4, swings=1, period=3)\n'
         'stage.caption("Longer period, stretched wave")'),
    ]),
    ("find all the primes below 50 with the sieve of Eratosthenes", [
        ("Cross out the multiples of each prime in turn",
         "Write the numbers 1 to 50. Keep 2 and cross out its multiples, then 3, then 5, then 7.",
         'stage.title("The sieve of Eratosthenes")\n'
         'grid = sieve_primes(stage, 50)\n'
         'stage.caption("What survives is prime")'),
        ("Only primes are left",
         "Everything left standing has no smaller factor: these are the primes.",
         'highlight(stage, grid)\n'
         'stage.equation("2, 3, 5, 7, 11, 13, \\ldots, 47", where="right")'),
    ]),
    ("why do you only need to sieve with primes up to the square root", [
        ("Sieve the numbers up to one hundred",
         "Up to 100 we only need 2, 3, 5 and 7, because 11 squared is already past 100.",
         'stage.title("Sieving to 100")\n'
         'grid = sieve_primes(stage, 100)\n'
         'stage.caption("Every composite below 100 has a factor below 10")'),
    ]),
    ("the primes thin out but they never stop", [
        ("The sieve up to sixty",
         "Among small numbers, primes are common.",
         'stage.title("Primes thin out")\n'
         'grid = sieve_primes(stage, 60)'),
        ("Thousands of primes in a spiral",
         "Plot thousands of them in polar coordinates: fewer and fewer per ring, but they keep coming.",
         'stage.clear()\n'
         'spiral = prime_spiral(stage, 2000)\n'
         'stage.caption("Rarer, but never gone")'),
    ]),
    ("how a ball rolling downhill finds the minimum of a cost function", [
        ("A bowl-shaped cost curve",
         "Here is a cost that is lowest at x equals one.",
         'stage.title("Downhill")\n'
         'ax = draw_axes(stage, x_range=(-3, 3), y_range=(0, 9))\n'
         'f = lambda x: (x - 1) ** 2 + 0.5\n'
         'g = plot_graph(stage, ax, f)'),
        ("A ball steps downhill by the slope",
         "Put a ball at the left and step it against the slope, again and again.",
         'ball = gradient_descent(stage, ax, f, -2.5, lr=0.3, steps=8)\n'
         'stage.caption("Each step follows the slope down")'),
        ("Too large a step overshoots",
         "Make the steps too big and the ball jumps from wall to wall.",
         'stage.clear(keep=[ax, g])\n'
         'ball2 = gradient_descent(stage, ax, f, -2.5, lr=0.95, steps=6, color=RED)\n'
         'stage.caption("Learning rate too large")'),
    ]),
    ("estimate the area under a curve with rectangles that get thinner", [
        ("A curve and the region under it",
         "Here is the curve x squared over three, from 0 to 3.",
         'stage.title("Area by rectangles")\n'
         'ax = draw_axes(stage, x_range=(0, 3), y_range=(0, 3))\n'
         'f = lambda x: x ** 2 / 3\n'
         'g = plot_graph(stage, ax, f)'),
        ("Rectangles under the curve, refined",
         "Fill it with rectangles, then halve their width, and halve it again.",
         'rects = riemann_refine(stage, ax, f, 0, 3)\n'
         'stage.caption("Thinner rectangles, closer fit")'),
        ("The exact area",
         "In the limit the rectangles fill exactly the area: the integral.",
         'stage.clear(keep=[ax, g])\n'
         'area = shade_area(stage, ax, f, 0, 3)\n'
         'stage.equation(r"\\int_0^3 \\frac{x^2}{3}\\,dx = 3", where="right")'),
    ]),
    ("the area under one arch of sin x is exactly 2", [
        ("Rectangles under one arch of the sine curve",
         "Split the arch from 0 to pi into strips and keep splitting.",
         'stage.title("One arch of sine")\n'
         'ax = draw_axes(stage, x_range=(0, 3.2), y_range=(0, 1.2))\n'
         'g = plot_graph(stage, ax, np.sin)\n'
         'rects = riemann_refine(stage, ax, np.sin, 0, PI)'),
        ("The total settles at two",
         "The sum of the strips settles at exactly two.",
         'stage.equation(r"\\int_0^\\pi \\sin x\\,dx = 2", where="right")'),
    ]),
    ("approximating e to the x near zero with polynomials", [
        ("The exponential curve",
         "Here is e to the x.",
         'stage.title("Taylor polynomials")\n'
         'ax = draw_axes(stage, x_range=(-2, 2), y_range=(-1, 7))\n'
         'g = plot_graph(stage, ax, np.exp, label="e^x")'),
        ("Polynomials with more terms hug the curve",
         "Add one term at a time: a constant, a line, a parabola, a cubic. Each hugs the curve longer.",
         'p = taylor_approximate(stage, ax, np.exp, 0, [1, 1, 1, 1, 1])\n'
         'stage.equation(r"e^x \\approx 1 + x + \\frac{x^2}{2} + \\frac{x^3}{6}", where="right")'),
    ]),
    ("multiplying a complex number by i is a quarter turn", [
        ("The complex plane with 1 and i",
         "Mark 1 and i on the complex plane.",
         'stage.title("Multiplying by i")\n'
         'cp = draw_complex_plane(stage)'),
        ("Multiplying by i turns everything a quarter turn",
         "Multiply by i: every point rotates ninety degrees about zero.",
         'multiply_complex(stage, cp, 1j)\n'
         'stage.caption("A quarter turn, no stretching")'),
        ("Twice is a half turn",
         "Do it twice and you have turned half way round: i times i is minus one.",
         'multiply_complex(stage, cp, 1j, points=[1j])\n'
         'stage.equation(r"i \\cdot i = -1", where="right")'),
    ]),
    ("Euler's formula: e to the i t walks around the unit circle", [
        ("A point e to the i t moving round the circle",
         "As t grows, e to the i t walks round the unit circle at speed one.",
         'stage.title("e^{it}")\n'
         'cp = draw_complex_plane(stage)\n'
         'walk = euler_circle(stage, cp)'),
        ("At t equals pi it reaches minus one",
         "Half way round, at t equals pi, it lands on minus one.",
         'stage.equation(r"e^{i\\pi} = -1", where="right")'),
    ]),
    ("what a shear transformation does to the plane", [
        ("A vector on the plane",
         "Start with the grid and one vector.",
         'stage.title("A shear")\n'
         'p = draw_plane(stage)\n'
         'v = draw_vector(stage, p, (1, 1), label="v")'),
        ("Apply the shear",
         "The shear slides each row sideways in proportion to its height.",
         'apply_matrix(stage, p, [[1, 1], [0, 1]], riders=[v])\n'
         'stage.caption("Rows slide; the x-axis stays put")'),
        ("Area is unchanged",
         "Squares become parallelograms of the same area: the determinant is one.",
         'stage.clear()\n'
         'p2 = draw_plane(stage)\n'
         'sq, lab = show_determinant(stage, p2, [[1, 1], [0, 1]])'),
    ]),
    ("a rotation by 90 degrees written as a matrix", [
        ("The basis vectors i-hat and j-hat",
         "Draw the two basis vectors.",
         'stage.title("Rotation as a matrix")\n'
         'p = draw_plane(stage)\n'
         'i_hat, j_hat = draw_basis(stage, p)'),
        ("Rotate the plane a quarter turn",
         "Rotate everything by ninety degrees: i-hat lands on j-hat, j-hat on minus i-hat.",
         'apply_matrix(stage, p, [[0, -1], [1, 0]], riders=[i_hat, j_hat])\n'
         'stage.equation(r"\\begin{bmatrix} 0 & -1 \\\\ 1 & 0 \\end{bmatrix}", where="right")'),
    ]),
    ("the directions the matrix [[3, 1], [0, 2]] only stretches", [
        ("Eigenvectors stay on their lines",
         "Most vectors turn under this matrix; two special directions only stretch.",
         'stage.title("Stretch directions")\n'
         'p = draw_plane(stage)\n'
         'show_eigenvectors(stage, p, [[3, 1], [0, 2]])\n'
         'stage.caption("Stretched by 3 and by 2, never turned")'),
    ]),
    ("rolling two dice: why 7 is the most likely total", [
        ("All 36 outcomes, the sevens lit",
         "Lay out all 36 ways two dice can land; six of them total seven.",
         'stage.title("Two dice")\n'
         'cells = draw_dice_grid(stage, highlight_sum=7)\n'
         'stage.caption("Six ways to make 7")'),
        ("Roll many times and the histogram peaks at 7",
         "Roll the pair hundreds of times: the totals pile up into a triangle peaked at seven.",
         'stage.clear()\n'
         'hist = grow_histogram(stage, lambda rng, k: rng.integers(1, 7, k) + rng.integers(1, 7, k), bins=np.arange(1.5, 13.5, 1))'),
    ]),
    ("the law of large numbers with coin flips", [
        ("The share of heads settles at one half",
         "Flip a coin again and again and track the fraction of heads.",
         'stage.title("Law of large numbers")\n'
         'flip_coins(stage, 200, 0.5)\n'
         'stage.caption("Wild at first, then it settles near 1/2")'),
    ]),
    ("the false positive paradox with a rare disease", [
        ("A population split by disease and test result",
         "One person in fifty has the disease; the test catches 95 percent and wrongly flags 5 percent of the healthy.",
         'stage.title("A rare disease")\n'
         'sq = bayes_square(stage, prior=0.02, sensitivity=0.95, false_pos=0.05)'),
        ("Most positives are healthy people",
         "Among everyone who tests positive, the healthy far outnumber the sick.",
         'stage.equation(r"P(D \\mid +) = \\frac{0.019}{0.019 + 0.049} \\approx 0.28", where="right")'),
    ]),
    ("two waves interfering: when they add up and when they cancel", [
        ("Two waves in step add up",
         "Two waves in step: peaks meet peaks and the sum is twice as tall.",
         'stage.title("Interference")\n'
         'ax = draw_axes(stage, x_range=(0, 8), y_range=(-2.5, 2.5))\n'
         'superpose_waves(stage, ax, [(1, 1, 1), (1, 1, 1)])'),
        ("Slightly different frequencies beat",
         "Change one frequency a little: the sum swells and fades in beats.",
         'stage.clear(keep=[ax])\n'
         'superpose_waves(stage, ax, [(1, 1, 1), (1, 1.3, 1.3)])\n'
         'stage.caption("Beats: in step, then out")'),
    ]),
    ("building a square wave out of sine waves", [
        ("Odd sines added one at a time",
         "Add sine x, then a third of sine 3x, then a fifth of sine 5x: corners appear.",
         'stage.title("A square wave from sines")\n'
         'ax = draw_axes(stage, x_range=(0, 6.3), y_range=(-1.5, 1.5))\n'
         'build_fourier_series(stage, ax, n_terms=7)\n'
         'stage.caption("More terms, sharper corners")'),
    ]),
    ("how a signal flows through a small neural network", [
        ("Layers of neurons joined by weights",
         "Three inputs, a hidden layer of four, two outputs: every neuron joined to the next layer.",
         'stage.title("A small network")\n'
         'net = draw_neural_net(stage, (3, 4, 2))'),
        ("The hidden layer lights up",
         "The hidden layer takes weighted sums of the inputs.",
         'highlight(stage, net.layers[1])\n'
         'stage.equation(r"a = \\sigma(Wx + b)", where="right")'),
    ]),
    ("bubble sort: swapping neighbours until the list is in order", [
        ("An unsorted array as bars",
         "Five numbers, out of order, drawn as bars.",
         'stage.title("Bubble sort")\n'
         'bars = draw_array(stage, [5, 2, 8, 1, 4])'),
        ("Swap neighbours that are out of order",
         "Compare neighbours and swap them when the left one is bigger.",
         'swap_bars(stage, bars, 0, 1)\n'
         'swap_bars(stage, bars, 2, 3)\n'
         'swap_bars(stage, bars, 3, 4)\n'
         'stage.caption("The largest bubbles to the end")'),
    ]),
    ("why 1/3 + 1/9 + 1/27 + ... adds up to one half", [
        ("Partial sums of powers of a third",
         "Add the terms one by one and watch the running total creep up.",
         'stage.title("A geometric series")\n'
         'bars, total = show_partial_sums(stage, lambda k: 1 / 3 ** k, 8)'),
        ("The limit",
         "The totals approach one half and never pass it.",
         'stage.equation(r"\\sum_{k=1}^{\\infty} \\frac{1}{3^k} = \\frac{1}{2}", where="right")'),
    ]),
]


_STOP = set("""a an the of and or to in on at by for with as is are be it its
this that why how what does do show showing shows from into one two using
use over under up down out every each any all about than then""".split())


def _words(text: str) -> set[str]:
    return {w[:6] for w in re.findall(r"[a-z]+", text.lower())
            if w not in _STOP and len(w) > 2}


def nearest(text: str, min_overlap: int = 1):
    """The hand-written scene for the same subject (the families' keyword
    test: its blocks belong to the text's best-matching subject) that shares
    the most words with ``text``; None if no scene is on that subject.

    Words alone matched "least squares" to the triangle-angles scene.
    """
    from forge.kit.families import BLOCK_FAMILY, FAMILIES, kit_calls
    t = text.lower()
    hits = {f: len(re.findall(words, t)) for f, (_, words) in FAMILIES.items()}
    top = max(hits, key=hits.get)
    if not hits[top]:
        return None
    want = _words(text)
    best, score = None, 0
    for req, spec in SCENES:
        fams = {BLOCK_FAMILY[b] for b, _ in kit_calls("\n".join(c for *_, c in spec))}
        if top not in fams:
            continue
        have = _words(req + " " + " ".join(i for i, _, _ in spec))
        s = len(want & have)
        if s > score:
            best, score = (req, spec), s
    return best if score >= min_overlap else None


def example(text: str, max_beats: int = 3) -> str:
    """A worked example for the coder's prompt, or ""."""
    got = nearest(text)
    if got is None:
        return ""
    req, spec = got
    out = [f"A WORKED EXAMPLE (a different request, for how blocks are used)",
           f"  request: {req}"]
    for k, (intent, _, code) in enumerate(spec[:max_beats]):
        out.append(f"  beat {k + 1}: {intent}")
        out += [f"    {l}" for l in code.splitlines()]
    return "\n".join(out)
