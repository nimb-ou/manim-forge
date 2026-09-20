"""An outcome is an answer about one version of the kernel, not forever.

Three separate times a run failed, I pushed a fixed kernel, and the
autopilot refused to look at it because a state file still held the previous
verdict — a manual reset in the middle of the thing whose purpose is not
needing one. These tests hold the fix to both halves: a stale outcome must
not block, and a current one must still stop a reboot from re-running a
ninety-minute evaluation over its own numbers.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ap(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "autopilot", ROOT / "scripts" / "autopilot.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["autopilot"] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "STATE_DIR", tmp_path)
    monkeypatch.setattr(mod, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(mod, "LOG", tmp_path / "autopilot.log")
    return mod


def test_fingerprint_tracks_the_kernel_source(ap, tmp_path, monkeypatch):
    src = tmp_path / "kaggle" / "01_sft.py"
    src.parent.mkdir(parents=True)
    src.write_text("print('a')\n")
    monkeypatch.setattr(ap, "ROOT", tmp_path)
    first = ap.kernel_fingerprint()
    src.write_text("print('b')\n")
    assert ap.kernel_fingerprint() != first


def test_a_missing_kernel_does_not_raise(ap, tmp_path, monkeypatch):
    """A fingerprint is a convenience; it must never be the thing that fails."""
    monkeypatch.setattr(ap, "ROOT", tmp_path)
    assert ap.kernel_fingerprint() == "missing"


def test_archive_preserves_the_previous_run(ap, tmp_path):
    prior = {"outcome": "kernel-failed", "updated": "2026-09-20T21:54:17"}
    (tmp_path / "kernel-output").mkdir()
    (tmp_path / "kernel-output" / "run.log").write_text("the diagnosis")
    ap.archive_state(prior)
    kept = list((tmp_path / "history").glob("state-*.json"))
    assert len(kept) == 1
    assert json.loads(kept[0].read_text())["outcome"] == "kernel-failed"
    moved = list((tmp_path / "history").glob("kernel-output-*"))
    assert moved and (moved[0] / "run.log").read_text() == "the diagnosis"
    assert not ap.STATE.exists()


def test_archive_survives_having_no_kernel_output(ap, tmp_path):
    ap.archive_state({"outcome": "timeout", "updated": "2026-09-20T10:00:00"})
    assert list((tmp_path / "history").glob("state-*.json"))
