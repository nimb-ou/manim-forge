"""Seed topics for synthetic generation.

Chosen for Nimit's stated domains — mathematics, geometry, vectors, and machine
learning — and biased toward concepts that *want* to move. "Prove the
Pythagorean theorem by rearrangement" is a better animation prompt than "define
a right triangle", because the explanation is the motion.

Each entry is (topic, difficulty). Difficulty drives how many beats the teacher
is asked for, which is how we get coverage from 3-second clips up to 60-second
explainers rather than everything landing at one length.
"""

GEOMETRY = [
    "Pythagorean theorem shown by rearranging four right triangles",
    "the area of a circle derived by unrolling it into triangles",
    "why the angles of a triangle sum to 180 degrees",
    "constructing the perpendicular bisector of a segment",
    "similar triangles and why their ratios are equal",
    "the locus of points equidistant from two fixed points",
    "tiling the plane with regular hexagons, and why pentagons fail",
    "inscribed angle theorem: the angle at the centre is twice the angle at the edge",
    "how a cone, cylinder and sphere of the same radius relate in volume",
    "unfolding a cube into its six-square net",
]

VECTORS = [
    "vector addition shown head to tail",
    "the dot product as projection of one vector onto another",
    "the cross product as the area of a parallelogram",
    "a linear transformation stretching and rotating the unit square",
    "eigenvectors as the directions that do not rotate under a transformation",
    "the determinant as the factor by which area scales",
    "span of two vectors sweeping out a plane",
    "linear independence shown by two vectors collapsing onto one line",
    "basis change: the same point described in two coordinate systems",
    "a vector field with arrows scaled by magnitude",
    "divergence shown as flow out of a small circle",
    "curl shown as rotation of a paddle wheel in a flow",
]

CALCULUS = [
    "the derivative as the slope of a tangent line approaching a point",
    "Riemann sums converging to the area under a curve",
    "the fundamental theorem of calculus linking area and slope",
    "Taylor series approximating sine with more and more terms",
    "a limit approaching a value from both sides",
    "the chain rule shown as nested rates of change",
    "optimisation: finding a maximum by where the slope is zero",
    "arc length approximated by shrinking straight segments",
]

PROBABILITY = [
    "the law of large numbers: coin flips converging to one half",
    "a binomial distribution emerging from a Galton board",
    "conditional probability shown with shrinking sample spaces",
    "Bayes theorem as reallocation of belief across a grid",
    "the birthday paradox: collision probability rising with group size",
    "expected value of a dice roll shown as a weighted balance",
    "random walk drifting over many steps",
    "sampling from a distribution shown as dots filling a histogram",
]

MACHINE_LEARNING = [
    "gradient descent rolling downhill on a loss surface",
    "a linear regression line fitting itself to scattered points",
    "a decision boundary separating two classes of points",
    "k-means clustering: centroids moving to the middle of their groups",
    "overfitting shown as a wiggly curve chasing every point",
    "a single neuron computing a weighted sum then a nonlinearity",
    "backpropagation as error flowing backwards through a small network",
    "the effect of learning rate: too small crawls, too large overshoots",
    "a convolution kernel sliding across an image grid",
    "principal component analysis finding the direction of most variance",
    "train/test split shown as partitioning a dataset",
    "the sigmoid squashing any input into zero-to-one",
]

NUMBER_THEORY = [
    "the sieve of Eratosthenes crossing out multiples",
    "Euclid's algorithm for greatest common divisor by repeated subtraction",
    "modular arithmetic shown as a clock face",
    "the Fibonacci sequence and the golden ratio spiral",
    "why the square root of two cannot be a fraction",
    "binary representation counting up in place values",
    "the collatz sequence bouncing toward one",
]

DISCRETE = [
    "a chessboard and the number of squares a rook attacks",
    "the handshake problem: counting pairs in a group",
    "a binary search halving the search space each step",
    "graph traversal visiting nodes breadth first",
    "the pigeonhole principle with more items than containers",
    "permutations versus combinations of a small set",
    "Pascal's triangle built row by row from sums above",
    "a sorting algorithm comparing and swapping bars",
]


PHYSICS = [
    "projectile motion tracing a parabola under gravity",
    "simple harmonic motion of a mass on a spring, with its sine trace",
    "a pendulum swinging, and why small angles make it simple",
    "conservation of momentum in an elastic collision of two balls",
    "the electric field of a point charge drawn as radiating arrows",
    "field lines between two opposite charges",
    "wave interference: two sources producing a pattern of peaks and troughs",
    "refraction: a light ray bending as it enters glass",
    "the Doppler effect as a moving source compresses wavefronts",
    "orbital motion as continuous falling around a planet",
    "torque as force applied at a distance from a pivot",
    "standing waves on a string at the first three harmonics",
    "the ideal gas law shown as particles in a shrinking box",
    "buoyancy: displaced volume versus weight",
]

CHEMISTRY = [
    "electron shells filling up around a nucleus",
    "a covalent bond as two atoms sharing an electron pair",
    "the shape of a water molecule and why it is bent",
    "sp3 hybridisation producing a tetrahedral arrangement",
    "balancing a chemical equation by counting atoms on each side",
    "titration: pH changing as base is added, with the curve drawn",
    "reaction rate as concentration decaying over time",
    "a benzene ring and its delocalised electrons",
    "ionic bonding as electron transfer between sodium and chlorine",
]

SHAPES_3D = [
    "a cube rotating in three dimensions, showing its hidden edges",
    "the five Platonic solids, one after another",
    "slicing a cone to produce circle, ellipse, parabola and hyperbola",
    "a sphere built from stacked circles of varying radius",
    "a torus traced by a circle swept around an axis",
    "unfolding a 3D solid into its 2D net",
    "a surface z = f(x, y) with its gradient field beneath",
    "projecting a 3D object onto a 2D plane as its shadow",
    "a helix as circular motion combined with steady rise",
]

GRAPHS = [
    "a graph traversed breadth first from a starting node",
    "depth first search exploring one branch to the end",
    "Dijkstra's algorithm relaxing edges toward the shortest path",
    "a minimum spanning tree growing edge by edge",
    "the seven bridges of Konigsberg and why the walk is impossible",
    "a bipartite graph and a maximum matching between its sides",
    "graph colouring with the fewest colours that work",
    "a cycle detected while traversing a directed graph",
    "degree distribution of a network shown as a histogram",
]

PUZZLES = [
    "the Monty Hall problem played out across many trials",
    "towers of Hanoi solved recursively",
    "the two-envelope paradox laid out with expected values",
    "a knight's tour covering every square of a chessboard",
    "the 100 prisoners problem and the loop strategy",
    "why 0.999... equals 1, shown geometrically",
    "the Tower of Hanoi step count doubling with each disc",
    "a magic square and why its rows, columns and diagonals agree",
    "the coupon collector problem: how long to collect every one",
]

ALL = (
    [(t, "geometry") for t in GEOMETRY]
    + [(t, "vectors") for t in VECTORS]
    + [(t, "calculus") for t in CALCULUS]
    + [(t, "probability") for t in PROBABILITY]
    + [(t, "machine-learning") for t in MACHINE_LEARNING]
    + [(t, "number-theory") for t in NUMBER_THEORY]
    + [(t, "discrete") for t in DISCRETE]
    + [(t, "physics") for t in PHYSICS]
    + [(t, "chemistry") for t in CHEMISTRY]
    + [(t, "shapes-3d") for t in SHAPES_3D]
    + [(t, "graphs") for t in GRAPHS]
    + [(t, "puzzles") for t in PUZZLES]
)

#: Beat counts requested per variation, so the corpus spans short clips through
#: full explainers rather than clustering at a single length.
LENGTHS = [
    ("short", 2, "a single idea, roughly 6 seconds"),
    ("medium", 5, "a short explanation, roughly 20 seconds"),
    ("long", 9, "a full explainer, roughly 45 seconds"),
]
