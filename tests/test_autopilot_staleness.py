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


def test_zero_byte_weights_are_not_a_collected_adapter(ap, tmp_path):
    """Run 17's failure: the download broke and left 0 bytes behind.

    The old check asked whether a file with a .safetensors suffix existed.
    One did. So the adapter was declared collected and handed to a converter
    that died on "Unable to read 8 bytes from file" — a truncated transfer
    reported as a conversion bug, an hour after the training that earned it.
    """
    found = tmp_path / "adapter"
    found.mkdir()
    (found / "adapter_config.json").write_text("{}")
    (found / "adapter_model.safetensors").write_bytes(b"")
    ok, detail = ap.weights_are_readable(found)
    assert not ok
    assert "0 bytes" in detail


def test_a_partial_download_is_not_a_collected_adapter(ap, tmp_path):
    found = tmp_path / "adapter"
    found.mkdir()
    (found / "adapter_model.safetensors").write_bytes(b"\x00" * 6_153_124)
    ok, detail = ap.weights_are_readable(found)
    assert not ok, detail


def test_config_without_weights_is_not_a_collected_adapter(ap, tmp_path):
    found = tmp_path / "adapter"
    found.mkdir()
    (found / "adapter_config.json").write_text("{}")
    ok, detail = ap.weights_are_readable(found)
    assert not ok
    assert "no adapter_model weights" in detail


def test_a_real_safetensors_file_reads_back(ap, tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    found = tmp_path / "adapter"
    found.mkdir()
    st.save_file({"lora_A": torch.zeros(4096, 64)},
                 str(found / "adapter_model.safetensors"))
    ok, detail = ap.weights_are_readable(found)
    assert ok, detail
    assert "MB" in detail


def test_a_small_sidecar_file_is_not_mistaken_for_weights(ap, tmp_path):
    """training_args.bin is 5.7 KB and belongs there.

    Matching on suffix alone rejected a good 161 MB adapter because a file
    called training_args.bin sat next to it. A validator with false
    positives is one you start overriding, which is how you end up with no
    validator.
    """
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    found = tmp_path / "adapter"
    found.mkdir()
    st.save_file({"lora_A": torch.zeros(4096, 64)},
                 str(found / "adapter_model.safetensors"))
    (found / "training_args.bin").write_bytes(b"\x00" * 5777)
    ok, detail = ap.weights_are_readable(found)
    assert ok, detail
