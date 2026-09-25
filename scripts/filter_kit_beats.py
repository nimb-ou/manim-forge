"""Keep the kit beats whose picture is about their beat, and is new.

The first full teacher scene I looked at was kit-shaped slideware: every
beat of an arc about Escher's Print Gallery drew the same plane with two
vectors under a different title. It renders and it "draws something", so
the visual-beat filter passed it; trained on, it teaches "draw vectors for
everything". Two filters, both deterministic (the LLM judges tried said YES
to "Escher's lithograph" drawn as two vectors):

  * relevance -- each kit family (linear algebra, calculus, waves, complex
    numbers, chance, numbers, geometry, networks and algorithms) has words
    that its pictures are about; a beat is kept only if a family it draws
    with matches the beat's own intent, narration or request. A beat
    drawing with raw Manim only is kept (this cannot judge it);
  * novelty -- a beat whose kit calls, with their literal arguments, repeat
    an earlier beat's in the same scene is the same picture again.

    ./.venv/bin/python scripts/filter_kit_beats.py
      -> data/kit/kit_beats_clean.jsonl
"""
from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = ROOT / "data" / "kit" / "kit_beats.jsonl"
OUT = ROOT / "data" / "kit" / "kit_beats_clean.jsonl"

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
         "riemann_refine", "trace_graph", "taylor_approximate",
         "narrow_epsilon_band", "gradient_descent", "diffuse_heat"},
        r"function|graph|curve|slope|derivative|tangent|integra|area|rate|"
        r"limit|approximat|taylor|polynomial|descent|minimi|heat|temperat|"
        r"plot|exponential|growth|decay|x\^|f\(x\)|calculus|chang|velocity|"
        r"accelerat|parabola|epsilon|delta|continu|cost|loss"),
    "waves": (
        {"animate_wave", "superpose_waves", "build_fourier_series",
         "circle_to_sine", "wind_signal"},
        r"wave|frequen|fourier|sine|cosine|oscillat|signal|sound|light|"
        r"period|vibrat|harmonic|interfer|spectrum"),
    "complex": (
        {"draw_complex_plane", "multiply_complex", "euler_circle"},
        r"complex|imaginar|euler|e\^|rotat|\bi\b|unit circle|phase"),
    "chance": (
        {"draw_dice_grid", "flip_coins", "grow_histogram", "bayes_square",
         "draw_bars"},
        r"probab|chance|random|dice|coin|distribut|bayes|test|sample|average|"
        r"histogram|data|normal|gaussian|binomial|likel|belief|odds|expect|"
        r"frequen"),
    "numbers": (
        {"draw_number_line", "mark_point", "fill_halving_squares",
         "prime_spiral", "show_partial_sums"},
        r"number|line|series|sum|half|prime|infinit|fraction|converg|diverg|"
        r"sequence|integer|count|zeta|harmonic"),
    "geometry": (
        {"slice_circle", "unroll_slices", "draw_right_triangle"},
        r"circle|area|triangle|pythag|\bpi\b|π|radius|circumference|"
        r"geometr|angle|hypotenuse"),
    "networks and algorithms": (
        {"draw_neural_net", "draw_network", "hanoi_moves", "bit_grid",
         "draw_array", "swap_bars", "convolve_bars", "flow_particles",
         "draw_vector_field"},
        r"neural|network|layer|neuron|node|edge|graph theory|hanoi|recurs|"
        r"\bbit|parity|code|error|sort|array|convolution|kernel|field|flow|"
        r"curl|divergence|fluid|weight|learn|algorithm|step"),
}
BLOCK_FAMILY = {b: f for f, (blocks, _) in FAMILIES.items() for b in blocks}


def beat_text(row: dict) -> str:
    user = row["messages"][1]["content"]
    got = re.findall(r"^\s*(?:intent|narration):\s*(.*)$", user, re.M)
    return (" ".join(got) + " " + row.get("request", "")).lower()


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


def body(row: dict) -> str:
    text = row["messages"][2]["content"]
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def main() -> int:
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    # The visual critic's verdicts (critic_kit_scenes.py), where it has run:
    # a beat whose frame a vision model judged not to show its idea is out.
    critic = ROOT / "data" / "kit" / "critic.jsonl"
    no = set()
    if critic.exists():
        for l in critic.read_text().splitlines():
            if l.strip():
                c = json.loads(l)
                no |= {(c["scene"], int(k)) for k, v in c["verdicts"].items()
                       if v == "NO"}
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r["meta"]["scene"]].append(r)
    why = Counter()
    kept = []
    for scene, rs in by_scene.items():
        rs.sort(key=lambda r: r["meta"]["index"])
        seen: set[tuple] = set()
        for r in rs:
            if (scene, r["meta"]["index"]) in no:
                why["dropped: the visual critic said no"] += 1
                continue
            calls = kit_calls(body(r))
            if not calls:
                kept.append(r)
                why["kept (raw Manim, not judged)"] += 1
                continue
            sig = tuple(sorted(calls))
            if sig in seen:
                why["dropped: repeats an earlier picture"] += 1
                continue
            seen.add(sig)
            text = beat_text(r)
            fams = {BLOCK_FAMILY[b] for b, _ in calls}
            if not any(re.search(FAMILIES[f][1], text) for f in fams):
                why["dropped: picture not about the beat"] += 1
                continue
            kept.append(r)
            why["kept"] += 1
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    print(f"{len(kept)} of {len(rows)} rows kept -> {OUT}")
    for k, v in why.most_common():
        print(f"  {k:40s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
