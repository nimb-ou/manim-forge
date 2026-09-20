"""Collect hand-authored gold scenes into training rows.

Gold scenes are the style anchor: few, deliberate, and weighted heavily at
training time. Each carries a natural-language prompt written the way a person
would actually ask for it, because that prompt is what the model learns to map
from.
"""
import json
from pathlib import Path

GOLD = [
    {
        "module": "forge/gold/convolution.py",
        "scene": "Convolution",
        "prompt": "show convolution as sliding a small window of weights along a "
                  "signal, and show that the same operation blurs, finds edges, "
                  "or sharpens depending only on what the three weights are",
        "tags": ['machine-learning', 'signals'],
    },
    {
        "module": "forge/gold/neuron.py",
        "scene": "Neuron",
        "prompt": "show what a single neuron does — weights, a sum, a bias and an "
                  "activation — and prove that without the activation two stacked "
                  "layers collapse into one",
        "tags": ['machine-learning', 'neural-networks'],
    },
    {
        "module": "forge/gold/gradient_descent_2d.py",
        "scene": "GradientDescent2D",
        "prompt": "show gradient descent on a stretched valley rather than a "
                  "bowl, and explain why a single learning rate cannot be both "
                  "stable across the steep direction and fast along the shallow "
                  "one",
        "tags": ['machine-learning', 'optimisation'],
    },
    {
        "module": "forge/gold/decision_boundary.py",
        "scene": "DecisionBoundary",
        "prompt": "show a perceptron learning a straight line that separates two "
                  "groups of points, moving only when it gets an example wrong, "
                  "then show that it never settles when the groups overlap",
        "tags": ['machine-learning', 'classification'],
    },
    {
        "module": "forge/gold/projectile.py",
        "scene": "Projectile",
        "prompt": "show that a projectile is two independent motions at once — "
                  "constant sideways drift and a ball thrown straight up — then "
                  "show that two complementary launch angles land in the same "
                  "place and forty-five degrees goes furthest",
        "tags": ['physics', 'motion'],
    },
    {
        "module": "forge/gold/chain_rule.py",
        "scene": "ChainRule",
        "prompt": "explain the chain rule as two machines feeding into each "
                  "other, work out the derivative of sine of three x plus one at "
                  "a point, and show what goes wrong if the outer derivative is "
                  "taken at the wrong place",
        "tags": ['calculus', 'derivatives'],
    },
    {
        "module": "forge/gold/modular_clock.py",
        "scene": "ModularClock",
        "prompt": "show arithmetic on a twelve-hour clock face, step around it by "
                  "five and by four, and explain why one visits every position "
                  "and the other only three",
        "tags": ['number-theory', 'modular'],
    },
    {
        "module": "forge/gold/bayes.py",
        "scene": "Bayes",
        "prompt": "show why a positive result on a very accurate medical test "
                  "still means you are probably fine, by drawing ten thousand "
                  "people as areas and counting the true and false positives",
        "tags": ['probability', 'bayes'],
    },
    {
        "module": "forge/gold/monty_hall.py",
        "scene": "MontyHall",
        "prompt": "play out the Monty Hall door problem, state the host's rule "
                  "explicitly, explain why switching wins two times in three, and "
                  "confirm it by simulating twenty thousand games",
        "tags": ['puzzles', 'probability'],
    },
    {
        "module": "forge/gold/irrational_sqrt2.py",
        "scene": "IrrationalSqrt2",
        "prompt": "prove that the square root of two cannot be written as a "
                  "fraction, by assuming it can and following the parity argument "
                  "to a contradiction",
        "tags": ['number-theory', 'proof'],
    },
    {
        "module": "forge/gold/euclid_gcd.py",
        "scene": "EuclidGcd",
        "prompt": "show Euclid's algorithm for the greatest common divisor as "
                  "cutting the largest possible squares off a rectangle until one "
                  "fits exactly",
        "tags": ['number-theory', 'algorithms'],
    },
    {
        "module": "forge/gold/recursion_hanoi.py",
        "scene": "TowerOfHanoi",
        "prompt": "solve the Tower of Hanoi with four discs and show that the "
                  "method is just the same problem one size smaller, then explain "
                  "why the number of moves is two to the n minus one",
        "tags": ['discrete', 'recursion'],
    },
    {
        "module": "forge/gold/binary_search.py",
        "scene": "BinarySearch",
        "prompt": "show binary search finding a number in a sorted list of "
                  "sixty-four by halving the range each look, and compare the "
                  "number of looks to checking one at a time",
        "tags": ['discrete', 'algorithms'],
    },
    {
        "module": "forge/gold/derivative_slope.py",
        "scene": "DerivativeSlope",
        "prompt": "show the derivative as a limit of slopes — draw a secant line "
                  "between two points on a curve, shrink the gap and show the "
                  "computed slopes settling on a single value, then draw the "
                  "tangent",
        "tags": ['calculus', 'plot', 'limits'],
    },
    {
        "module": "forge/gold/taylor.py",
        "scene": "Taylor",
        "prompt": "approximate the sine curve with polynomials, adding one term "
                  "at a time and showing the error shrink near the centre, then "
                  "show the same polynomial diverging far from it",
        "tags": ['calculus', 'plot', 'series'],
    },
    {
        "module": "forge/gold/pythagoras.py",
        "scene": "Pythagoras",
        "prompt": "prove Pythagoras by rearrangement — pack four copies of the "
                  "same right triangle into a square two different ways and show "
                  "the leftover areas must be equal",
        "tags": ['geometry', 'proof'],
    },
    {
        "module": "forge/gold/circle_area.py",
        "scene": "CircleArea",
        "prompt": "show why the area of a circle is pi r squared by cutting it "
                  "into wedges and interleaving them into a rectangle, and show "
                  "the approximation improving as the wedges get thinner",
        "tags": ['geometry', 'limits'],
    },
    {
        "module": "forge/gold/electron_shells.py",
        "scene": "ElectronShells",
        "prompt": "show electrons filling shells around a nucleus and why the "
                  "shells hold two, eight and eighteen — derive the capacities by "
                  "counting orbitals rather than listing them",
        "tags": ['chemistry', 'atoms'],
    },
    {
        "module": "forge/gold/covalent_bond.py",
        "scene": "CovalentBond",
        "prompt": "show a covalent bond forming as two hydrogen atoms come "
                  "together, plot the energy against their separation, and show "
                  "that the bond is the bottom of that curve",
        "tags": ['chemistry', 'energy', 'plot'],
    },
    {
        "module": "forge/gold/reaction_rate.py",
        "scene": "ReactionRate",
        "prompt": "show a reaction as individual particles reacting at random, "
                  "then plot the concentration falling and show it follows an "
                  "exponential with a fixed half-life",
        "tags": ['chemistry', 'kinetics', 'plot'],
    },
    {
        "module": "forge/gold/bfs_graph.py",
        "scene": "BreadthFirst",
        "prompt": "show breadth-first search expanding outward from a node in "
                  "waves, and show that it finds the shortest path when every "
                  "edge costs one",
        "tags": ['graphs', 'algorithms'],
    },
    {
        "module": "forge/gold/dijkstra.py",
        "scene": "Dijkstra",
        "prompt": "show Dijkstra's algorithm relaxing edges and settling on the "
                  "cheapest route, on a graph where the route with fewest steps "
                  "is not the cheapest, and explain why the greedy choice is safe",
        "tags": ['graphs', 'algorithms'],
    },
    {
        "module": "forge/gold/graph_colouring.py",
        "scene": "GraphColouring",
        "prompt": "colour a graph so no two connected nodes match, show that the "
                  "greedy method's answer depends on the order you take the nodes "
                  "in, and compare it to the true minimum",
        "tags": ['graphs', 'algorithms', 'complexity'],
    },
    {
        "module": "forge/gold/konigsberg.py",
        "scene": "Konigsberg",
        "prompt": "show the seven bridges of Konigsberg, try and fail to cross "
                  "each exactly once, then prove it is impossible by counting how "
                  "many bridges touch each piece of land",
        "tags": ['graphs', 'proof', 'history'],
    },
    {
        "module": "forge/gold/rook_corners.py",
        "scene": "RookCorners",
        "prompt": "build a chess grid, add rooks on it and animate the rook's "
                  "possible movements, then calculate the probability that a "
                  "randomly placed rook is on a corner square",
        "tags": ["discrete", "probability", "grid"],
    },
    {
        "module": "forge/gold/gradient_descent.py",
        "scene": "LearningRate",
        "prompt": "explain how the learning rate affects gradient descent — "
                  "show a ball rolling down a loss curve with a good rate, "
                  "then one that is too large and overshoots, then one that is "
                  "too small and never arrives",
        "tags": ["machine-learning", "optimisation", "plot"],
    },
    {
        "module": "forge/gold/sphere_cube.py",
        "scene": "SphereInCube",
        "prompt": "show a sphere in 3D built out of dots, fit the largest cube "
                  "that can fit inside it, and calculate how much volume is "
                  "left over between the cube and the sphere",
        "tags": ["shapes-3d", "geometry", "volume"],
    },
    {
        "module": "forge/gold/riemann.py",
        "scene": "RiemannConvergence",
        "prompt": "show how Riemann sums converge to the area under a curve — "
                  "start with a couple of rectangles, keep doubling them, and "
                  "show the total approaching the exact integral, then show "
                  "that sampling the midpoint converges much faster",
        "tags": ["calculus", "convergence", "plot"],
    },
    {
        "module": "forge/gold/spring.py",
        "scene": "SpringTrace",
        "prompt": "show a mass on a spring oscillating, trace its position "
                  "over time to reveal a sine wave, show the energy moving "
                  "between kinetic and potential while the total stays "
                  "constant, and show how mass and stiffness change the period",
        "tags": ["physics", "oscillation", "energy"],
    },
    {
        "module": "forge/gold/sieve.py",
        "scene": "Sieve",
        "prompt": "show the sieve of Eratosthenes on the numbers up to a "
                  "hundred — cross out multiples of two, then three, five and "
                  "seven, and show that what survives is exactly the primes",
        "tags": ["number-theory", "grid", "algorithm"],
    },
    {
        "module": "forge/gold/law_large_numbers.py",
        "scene": "LawOfLargeNumbers",
        "prompt": "show the law of large numbers with coin flips — start with "
                  "ten flips landing far from a half, then keep flipping and "
                  "show the proportion wandering in toward it, and show that "
                  "the error shrinks like one over root n",
        "tags": ["probability", "convergence", "statistics"],
    },
    {
        "module": "forge/gold/number_line.py",
        "scene": "NumberLine1D",
        "prompt": "show a number line, place the integers then fractions then "
                  "an irrational number on it, and show that between any two "
                  "points there is always another",
        "tags": ["foundations", "number-theory"],
    },
    {
        "module": "forge/gold/area_rearranged.py",
        "scene": "AreaRearranged",
        "prompt": "show that cutting a shape and rearranging the pieces leaves "
                  "the area unchanged, and use that to turn a parallelogram "
                  "into a rectangle",
        "tags": ["foundations", "geometry", "area"],
    },
    {
        "module": "forge/gold/function_machine.py",
        "scene": "FunctionMachine",
        "prompt": "show a function as a machine taking an input and producing "
                  "an output, then show the same function as a curve, and "
                  "connect the two views",
        "tags": ["foundations", "calculus"],
    },
    {
        "module": "forge/gold/coordinate_plane.py",
        "scene": "CoordinatePlane",
        "prompt": "build the coordinate plane from two number lines, show how a "
                  "pair of numbers names exactly one point and that the order "
                  "matters, work out the distance between two points with a "
                  "right triangle, then show that a rule like x squared plus y "
                  "squared equals 25 traces out a circle",
        "tags": ["foundations", "geometry", "plot"],
    },
    {
        "module": "forge/gold/angle_sum.py",
        "scene": "AngleSum",
        "prompt": "show why the three angles of any triangle add to a straight "
                  "line, by tearing off the corners and fitting them together",
        "tags": ["foundations", "geometry"],
    },
    {
        "module": "forge/gold/dot_product.py",
        "scene": "DotProduct",
        "prompt": "explain the dot product as a projection — show two vectors, "
                  "drop a perpendicular so one casts a shadow on the other, and "
                  "show that the shadow times the length gives the dot product, "
                  "then swing past ninety degrees so the sign goes negative",
        "tags": ["vectors", "geometry", "projection"],
    },
]

out = Path("data/gold/gold.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for i, g in enumerate(GOLD):
        code = Path(g["module"]).read_text()
        f.write(json.dumps({
            "prompt": g["prompt"],
            "code": code,
            "meta": {"id": f"gold:{i:03d}", "source": "gold",
                     "scene": g["scene"], "tags": g["tags"],
                     "n_play_calls": code.count("self.play(")},
        }) + "\n")
print(f"{len(GOLD)} gold scenes -> {out}")
for g in GOLD:
    c = Path(g['module']).read_text()
    print(f"  {g['scene']:<16} {c.count('@beat')} beats, {c.count('self.play(')} play calls")
