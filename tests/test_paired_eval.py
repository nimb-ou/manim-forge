"""The paired view has to separate a generation change from a repair change.

Those two failures look identical in the headline rate and call for
completely different work — new training data versus a different kind of
training row — so a comparison that cannot tell them apart is not worth
running.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "paired_eval.py"


def trial(i, ok, history):
    return {"index": i, "ok": ok, "rounds": len(history) - 1,
            "history": history, "final_error": history[-1], "lint": []}


def run(tmp_path, control, tuned):
    (tmp_path / "c.json").write_text(json.dumps(control))
    (tmp_path / "t.json").write_text(json.dumps(tuned))
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--control", str(tmp_path / "c.json"),
         "--tuned", str(tmp_path / "t.json")], capture_output=True, text=True)


def test_repair_regression_is_named_as_one(tmp_path):
    """Same first error, control recovers, tuned burns its rounds."""
    control = [trial(i, True, ["api_misuse", "none"]) for i in range(10)]
    tuned = [trial(i, False, ["api_misuse"] * 5) for i in range(10)]
    r = run(tmp_path, control, tuned)
    assert r.returncode == 0, r.stderr
    assert "tuning LOST 10" in r.stdout
    assert "10 share the control's first-round error" in r.stdout
    assert "repair accounts for 100% of the difference" in r.stdout


def test_generation_change_is_attributed_to_generation(tmp_path):
    """Control passes first try; tuned fails outright. Repair explains none.

    The fixture matters: an earlier version had the control passing only
    *after* repair, so its first-try count was zero and the assertion about
    first-try could never hold. A test whose data cannot exhibit the thing
    it asserts passes or fails for the wrong reason.
    """
    control = [trial(i, True, ["none"]) for i in range(10)]
    tuned = [trial(i, False, ["syntax"] * 5) for i in range(10)]
    r = run(tmp_path, control, tuned)
    assert r.returncode == 0, r.stderr
    assert "first-try 10 -> 0" in r.stdout
    assert "repair accounts for 0% of the difference" in r.stdout


def test_both_mechanisms_are_reported_not_one(tmp_path):
    """The real run was both: -4 generation and -12 repair.

    A binary verdict called that 'generation changed' and was wrong about
    which dominated.
    """
    control = ([trial(i, True, ["none"]) for i in range(8)]
               + [trial(8 + i, True, ["api_misuse", "none"]) for i in range(4)])
    tuned = ([trial(i, True, ["none"]) for i in range(6)]
             + [trial(6 + i, False, ["api_misuse"] * 5) for i in range(6)])
    r = run(tmp_path, control, tuned)
    assert "first-try 8 -> 6 (-2)" in r.stdout
    assert "rescued   4 -> 0 (-4)" in r.stdout
    assert "repair accounts for 67% of the difference" in r.stdout


def test_an_improvement_is_not_reported_as_a_loss(tmp_path):
    control = [trial(i, False, ["api_misuse"] * 5) for i in range(10)]
    tuned = [trial(i, True, ["none"]) for i in range(10)]
    r = run(tmp_path, control, tuned)
    assert "tuning LOST 0" in r.stdout
    assert "tuning WON 10" in r.stdout


def test_mismatched_benchmarks_are_refused(tmp_path):
    control = [trial(i, True, ["none"]) for i in range(3)]
    tuned = [{"index": 900 + i, "ok": True, "rounds": 0, "history": ["none"]}
             for i in range(3)]
    r = run(tmp_path, control, tuned)
    assert r.returncode == 1
    assert "not the same benchmark" in r.stdout


def test_two_recorded_runs_of_the_same_model_agree_on_generation():
    """A property worth pinning: same model and prompts, same first errors.

    data/bench/escalate100 and rounds4 differ only in repair budget. Their
    first-round error mixes are identical, which is what makes a difference
    in the tuned run's mix attributable to the adapter rather than to
    sampling. If this ever fails, paired comparison here is unsound.
    """
    a = ROOT / "data" / "bench" / "escalate100_n100_r2.json"
    b = ROOT / "data" / "bench" / "rounds4_n100_r4.json"
    if not (a.exists() and b.exists()):
        import pytest
        pytest.skip("baseline artefacts not present")
    fa = [(t["index"], (t.get("history") or ["?"])[0])
          for t in json.loads(a.read_text())]
    fb = [(t["index"], (t.get("history") or ["?"])[0])
          for t in json.loads(b.read_text())]
    assert fa == fb
