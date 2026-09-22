"""A transposed conversion must fail this, or it proves nothing.

peft_to_mlx.py --verify proves the adapter *loads*. This proves it applies
the *same delta*, which is a different claim: a transposed conversion is
loaded, non-zero, passes the probe, and quietly degrades the model — and its
damage would be read as a finding about the corpus rather than about the
converter. So the test that matters is the one where the conversion is
wrong on purpose.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_adapter_math.py"

np = pytest.importorskip("numpy")
st = pytest.importorskip("safetensors.numpy")


def build(tmp_path: Path, transpose_bug: bool = False, r: int = 8,
          alpha: int = 16, n_in: int = 32, n_out: int = 24):
    rng = np.random.default_rng(0)
    peft, mlx = tmp_path / "peft", tmp_path / "mlx"
    peft.mkdir(), mlx.mkdir()
    (peft / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": alpha}))

    p_tensors, m_tensors = {}, {}
    for layer in (0, 1):
        A = rng.normal(0, 0.02, (r, n_in)).astype(np.float32)
        B = rng.normal(0, 0.02, (n_out, r)).astype(np.float32)
        pre = f"base_model.model.model.layers.{layer}.self_attn.q_proj"
        p_tensors[f"{pre}.lora_A.weight"] = A
        p_tensors[f"{pre}.lora_B.weight"] = B
        mre = f"model.layers.{layer}.self_attn.q_proj"
        if transpose_bug:
            # The plausible mistake: carry the PEFT orientation across
            # unchanged. Shapes still line up when in and out are swapped,
            # which is why it survives a shape check.
            m_tensors[f"{mre}.lora_a"] = A.T.copy()
            m_tensors[f"{mre}.lora_b"] = B.T.copy()[::-1].copy()
        else:
            m_tensors[f"{mre}.lora_a"] = A.T.copy()
            m_tensors[f"{mre}.lora_b"] = B.T.copy()
    st.save_file(p_tensors, str(peft / "adapter_model.safetensors"))
    st.save_file(m_tensors, str(mlx / "adapters.safetensors"))
    return peft, mlx


def run(peft: Path, mlx: Path):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--peft", str(peft), "--mlx", str(mlx),
         "--sample", "0"], capture_output=True, text=True)


def test_a_correct_conversion_passes(tmp_path):
    r = run(*build(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "applies the same delta" in r.stdout


def test_a_scrambled_conversion_fails(tmp_path):
    """If this ever passes, the checker is decorative."""
    r = run(*build(tmp_path, transpose_bug=True))
    assert r.returncode == 1, r.stdout
    assert "does NOT apply the same delta" in r.stdout


def test_a_missing_module_is_reported_not_ignored(tmp_path):
    peft, mlx = build(tmp_path)
    import safetensors.numpy as sn
    kept = {k: v for k, v in sn.load_file(str(mlx / "adapters.safetensors")).items()
            if "layers.1." not in k}
    sn.save_file(kept, str(mlx / "adapters.safetensors"))
    r = run(peft, mlx)
    assert r.returncode == 1
    assert "missing from one side" in r.stdout


def test_the_real_converted_adapter_is_faithful():
    peft = ROOT / "adapters" / "kaggle-sft" / "adapter"
    mlx = ROOT / "adapters" / "mlx-sft"
    if not (peft / "adapter_model.safetensors").exists() or \
            not (mlx / "adapters.safetensors").exists():
        pytest.skip("no converted adapter on disk")
    r = run(peft, mlx)
    assert r.returncode == 0, r.stdout
