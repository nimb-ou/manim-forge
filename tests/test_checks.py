"""On-screen arithmetic checks (forge/app/checks.py)."""
from forge.app.checks import arithmetic_errors, chain_errors


def test_right_arithmetic_passes():
    for s in ("80 - 12 = 68", r"\frac{9 \times 8}{2} = 36", "15% of £80 = £12" .replace(" of ", " × "),
              "1/3 = 0.33", r"4 \times 180^\circ = 720^\circ".replace("^\\circ", ""),
              "3,400 + 600 = 4,000", r"A = \frac{1}{2} \cdot 7 \cdot 4 = 14",
              r"15\% = 4 + 2", "1101 = -8 + 4 + 0 + 1"):
        assert chain_errors(s) == [], s


def test_slips_are_caught():
    assert chain_errors("80 × 0.85 = 72.8")
    assert chain_errors(r"\frac{9 \times 8}{2} = 72")
    assert chain_errors("8 + 7 + 6 + 5 + 4 + 3 + 2 + 1 = 35")


def test_algebra_and_inequalities_are_skipped():
    for s in ("a^2 + b^2 = c^2", "2x + 1 = 7", "x < 3", "pi r^2 = 3.14 r^2", "y = 3x"):
        assert chain_errors(s) == [], s


def test_scene_code_strings_are_found():
    code = ('stage.title("Handshakes")\n'
            'stage.equation(r"\\frac{9 \\times 8}{2} = 36", "8 + 7 = 16", where="right")\n'
            'x = "1 + 1 = 3"\n')
    assert arithmetic_errors(code) == ["8 + 7 = 16"]
