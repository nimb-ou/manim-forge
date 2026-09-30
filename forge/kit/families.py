"""Which kit blocks belong to which subject, and the words of each subject.

One definition for the training-data filter (filter_kit_beats.py), the
scorecard's relevance share (rescore.py) and the GRPO reward: a kit picture
counts only if its family's words appear in the beat it illustrates. GRPO v2
drew a plane and one vector for a neural network, Bayes' theorem and
backpropagation; each of those is a linear-algebra picture on a beat with
no linear-algebra words.
"""
from __future__ import annotations

import ast
import re

FAMILIES = {
    "linear algebra": (
        {"draw_plane", "draw_vector", "draw_basis", "apply_matrix",
         "draw_unit_square", "draw_span", "scale_vector", "show_determinant",
         "show_eigenvectors", "project_vector", "basis_grid"},
        r"vector|matri|linear|basis|span|plane|grid|transform|determinant|"
        r"eigen|dot product|projection|column|coordinat|shear|rotat|dimension|"
        r"scal|arrow|direction|space|cramer|cross product|inverse"),
    "calculus": (
        {"draw_axes", "plot_graph", "slide_tangent", "shade_area",
         "riemann_refine", "trace_graph", "taylor_approximate", "secant_to_tangent",
         "narrow_epsilon_band", "gradient_descent", "diffuse_heat"},
        r"function|graph|curve|slope|derivative|tangent|integra|area|rate|"
        r"limit|approximat|taylor|polynomial|descent|minimi|heat|temperat|"
        r"plot|exponential|growth|decay|x\^|f\(x\)|calculus|chang|velocity|"
        r"accelerat|parabola|epsilon|delta|continu|cost|loss"),
    "waves": (
        {"animate_wave", "superpose_waves", "build_fourier_series",
         "circle_to_sine", "wind_signal", "swing_pendulum"},
        r"wave|frequen|fourier|sine|cosine|oscillat|signal|sound|light|"
        r"period|vibrat|harmonic|interfer|spectrum|pendulum|swing"),
    "complex": (
        {"draw_complex_plane", "multiply_complex", "euler_circle"},
        r"complex|imaginar|euler|e\^|rotat|\bi\b|unit circle|phase"),
    "chance": (
        {"draw_dice_grid", "flip_coins", "grow_histogram", "bayes_square",
         "draw_bars", "plot_points"},
        r"probab|chance|random|dice|coin|distribut|bayes|test|sample|average|"
        r"histogram|data|normal|gaussian|binomial|likel|belief|odds|expect|"
        r"frequen"),
    "numbers": (
        {"draw_number_line", "mark_point", "fill_halving_squares",
         "prime_spiral", "show_partial_sums", "count_binary", "sieve_primes"},
        r"number|line|series|sum|half|prime|infinit|fraction|converg|diverg|"
        r"sequence|integer|count|zeta|harmonic|binary|digit|base|divisib|sieve"),
    "geometry": (
        {"slice_circle", "unroll_slices", "draw_right_triangle",
         "squares_on_sides", "angle_sum"},
        r"circle|area|triangle|pythag|\bpi\b|π|radius|circumference|"
        r"geometr|angle|hypotenuse|square|180|degree"),
    "networks and algorithms": (
        {"draw_neural_net", "draw_network", "hanoi_moves", "bit_grid",
         "draw_array", "swap_bars", "convolve_bars", "flow_particles",
         "draw_vector_field"},
        r"neural|network|layer|neuron|node|edge|graph theory|hanoi|recurs|"
        r"\bbit|parity|code|error|sort|array|convolution|kernel|field|flow|"
        r"curl|divergence|fluid|weight|learn|algorithm|step"),
}
BLOCK_FAMILY = {b: f for f, (blocks, _) in FAMILIES.items() for b in blocks}


def kit_calls(code: str) -> list[tuple[str, str]]:
    """(block, literal args) for each kit call, for relevance and novelty."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                and n.func.id in BLOCK_FAMILY:
            args = [ast.unparse(a) for a in n.args[1:]
                    if isinstance(a, (ast.Constant, ast.Tuple, ast.List,
                                      ast.UnaryOp))]
            out.append((n.func.id, "|".join(args)))
    return out


_SCAFFOLD = {"draw_plane", "draw_axes", "draw_number_line", "draw_complex_plane"}


def relevant(code: str, text: str) -> bool | None:
    """True if a kit family the code draws with matches ``text`` (a beat's
    intent and request, lower-cased); False if it draws only with families
    that do not; None if it uses no kit block (raw Manim: not judged)."""
    calls = {b for b, _ in kit_calls(code)}
    fams = {BLOCK_FAMILY[b] for b in calls}
    if not fams:
        return None
    # Axes, a grid or a number line and nothing on them is a background,
    # not the beat's picture -- the commonest held-out failure (2026-09-29:
    # "axes with the curve N = 64 * 2^-t" came back as empty axes, and
    # draw_axes' family matched the word "curve").
    if calls <= _SCAFFOLD and not re.search(r"\.(plot|animate)\b|\bplay\(|\b(Dot|Line|Arrow|Polygon)\(", code):
        return False
    return any(re.search(FAMILIES[f][1], text.lower()) for f in fams)


def hint(text: str, signatures: bool = False) -> str:
    """A line naming the kit blocks whose subject's words appear in ``text``
    (the request and the beat), for the coder's prompt; "" if none match."""
    t = text.lower()
    # Most-matched subject first, so its blocks survive the cut: "the
    # Pythagorean theorem with squares" also matches linear algebra ("plane"),
    # and alphabetical order dropped squares_on_sides.
    hits = {f: len(re.findall(words, t)) for f, (_, words) in FAMILIES.items()}
    fams = sorted((f for f in hits if hits[f]), key=lambda f: -hits[f])
    if not fams:
        return ""
    names = [b for f in fams for b in sorted(FAMILIES[f][0])]
    if not signatures:
        return "Blocks that draw this subject: " + ", ".join(names[:24]) + "."
    # With how to call each: kit v7 knew the new blocks' names from the
    # hint and guessed their arguments -- swing_pendulum(stage, p, m,
    # angle=30), superpose_waves(stage, ax, [g1, g2]) -- and crashed.
    sig = _signatures()
    lines = [sig.get(b, f"{b}(stage, ...)") for b in names[:14]]
    return "Blocks that draw this subject:\n" + "\n".join(f"  {l}" for l in lines)


_SIG: dict[str, str] = {}
_QUIET = {"color", "run_time", "where", "colors", "seed", "labels"}


def _signatures() -> dict[str, str]:
    """block -> how to call it: required arguments and the defaults worth
    knowing, then the first line of its docstring."""
    if not _SIG:
        import inspect
        from forge.kit import kit
        for name in sorted(kit.KIT_BLOCKS):
            fn = getattr(kit, name)
            parts = []
            for q in inspect.signature(fn).parameters.values():
                if q.default is inspect.Parameter.empty:
                    parts.append(q.name.rstrip("_"))
                elif q.name not in _QUIET and len(parts) < 6:
                    parts.append(f"{q.name}={q.default!r}")
            doc = (inspect.getdoc(fn) or "").split("\n")[0].rstrip(".")
            _SIG[name] = f"{name}({', '.join(parts)})" + (f"  -- {doc}" if doc else "")
    return _SIG
