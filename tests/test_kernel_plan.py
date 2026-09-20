"""The kernel's retry plan has to terminate, and has to move the right lever.

The plan is what makes the kernel autonomous, so it is also the thing that
turns one bad decision into a wasted twelve-hour session. These tests pull
`next_attempt` out of the kernel source and exercise it directly, because
the alternative way to discover that it loops forever is a GPU session.
"""
import ast
from pathlib import Path

import pytest

KERNEL = Path(__file__).resolve().parents[1] / "kaggle" / "01_sft.py"


@pytest.fixture(scope="module")
def next_attempt():
    """Extract the plan function from the kernel without running the kernel."""
    tree = ast.parse(KERNEL.read_text())
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "next_attempt"),
              None)
    assert fn is not None, "next_attempt is no longer a module-level function"
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<plan>", "exec"), ns)
    return ns["next_attempt"]


def test_oom_shortens_the_sequence_and_keeps_amp(next_attempt):
    """An OOM is answered with less sequence, not with fp32 activations."""
    assert next_attempt(True, 2048, "oom") == (True, 1536)


def test_oom_does_not_descend_past_1536(next_attempt):
    """1024 truncates 56% of the corpus; a half-scene adapter is not a result."""
    assert next_attempt(True, 1536, "oom") is None
    assert next_attempt(False, 1536, "oom") is None


def test_non_finite_turns_amp_off_and_keeps_the_sequence(next_attempt):
    assert next_attempt(True, 2048, "non-finite") == (False, 2048)
    assert next_attempt(True, 1536, "non-finite") == (False, 1536)


def test_non_finite_without_amp_has_nothing_left(next_attempt):
    """AMP was the hypothesis; with it off, another run tests nothing."""
    assert next_attempt(False, 2048, "non-finite") is None


def test_the_plan_terminates_from_every_start(next_attempt):
    """Walk every reachable path; none may revisit a state or run away."""
    for start in [(True, 2048), (False, 2048), (True, 1536), (False, 1536)]:
        for why in ("oom", "non-finite"):
            seen, attempt, steps = set(), start, 0
            while attempt is not None and attempt not in seen:
                seen.add(attempt)
                attempt = next_attempt(attempt[0], attempt[1], why)
                steps += 1
                assert steps < 10, f"{start}/{why} does not terminate"


def test_the_loop_guards_against_repeating_a_combination(next_attempt):
    """The driver's `while` must check membership, not just None."""
    src = KERNEL.read_text()
    assert "while _attempt is not None and _attempt not in _seen:" in src
    assert "_seen.add(_attempt)" in src
