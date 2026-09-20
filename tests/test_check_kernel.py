"""The checker has to still catch the bugs it was built for.

A static checker that stops reporting anything is indistinguishable from one
that works, so each historical failure gets a test. The last two entries are
the false positives that widening it introduced, kept so narrowing it again
would be visible.
"""
import subprocess
import sys
from pathlib import Path

CHECK = Path(__file__).resolve().parents[1] / "scripts" / "check_kernel.py"


def run(tmp_path: Path, src: str) -> subprocess.CompletedProcess:
    f = tmp_path / "kernel.py"
    f.write_text(src)
    return subprocess.run([sys.executable, str(CHECK), str(f)],
                          capture_output=True, text=True)


def test_class_base_imported_later_is_caught(tmp_path):
    """Smoke run 1: `class G(TrainerCallback)` above the import of it."""
    r = run(tmp_path, "class G(TrainerCallback):\n"
                      "    pass\n"
                      "from transformers import TrainerCallback\n")
    assert r.returncode == 1
    assert "TrainerCallback" in r.stdout


def test_module_level_use_before_assignment_is_caught(tmp_path):
    """Run 9: a patch landed in a comment, leaving the use above the bind."""
    r = run(tmp_path, "print(trainer)\ntrainer = 1\n")
    assert r.returncode == 1
    assert "trainer" in r.stdout


def test_name_that_is_only_a_function_local_is_caught(tmp_path):
    """The `tok` defect: wrapping a block made a global into a local."""
    r = run(tmp_path, "def build():\n"
                      "    tok = 1\n"
                      "    return tok\n"
                      "build()\n"
                      "print(tok)\n")
    assert r.returncode == 1
    assert "tok" in r.stdout


def test_closure_over_enclosing_scope_is_not_a_problem(tmp_path):
    """A nested class's method may read anything its enclosing body binds."""
    r = run(tmp_path, "import torch\n"
                      "def outer():\n"
                      "    def helper(x):\n"
                      "        return x\n"
                      "    class Inner:\n"
                      "        def go(self, v):\n"
                      "            return helper(v) + Inner.K\n"
                      "        K = 2\n"
                      "    return Inner\n")
    assert r.returncode == 0, r.stdout


def test_forward_reference_from_a_function_is_not_a_problem(tmp_path):
    """A module function may call something defined below it."""
    r = run(tmp_path, "def a():\n    return b()\n\ndef b():\n    return 1\n")
    assert r.returncode == 0, r.stdout


def test_the_real_kernel_is_clean(tmp_path):
    kernel = Path(__file__).resolve().parents[1] / "kaggle" / "01_sft.py"
    r = subprocess.run([sys.executable, str(CHECK), str(kernel)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
