"""The Kaggle training path, checked in the form it actually runs.

Two failures motivated this file, and both were cheap to have caught:

1. `kaggle/01_sft.py` is pushed as kernel_type "script" -- plain Python, where
   IPython's `!pip` is a SyntaxError. The first run died on line 19. I had
   run ast.parse on this file earlier the same day and *stripped the magics
   first to make it pass*, which silenced the only check that would have
   told me the truth.

2. The workflow classified Kaggle's status with a case-sensitive glob over
   the whole message. Kaggle reports `KernelWorkerStatus.ERROR` in capitals,
   so the poller could not see a failure and would have spent nine hours
   waiting on a job that was already dead.

So: parse the script with no preprocessing, and run the workflow's own shell
rather than a Python imitation of it.
"""
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle" / "01_sft.py"
WORKFLOW = ROOT / ".github" / "workflows" / "train.yml"

#: The authoritative list, read from kagglesdk.kernels.types.kernels_enums
#: rather than remembered.
STATUSES = ["QUEUED", "RUNNING", "COMPLETE", "ERROR",
            "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED", "NEW_SCRIPT"]
TERMINAL_OK = {"COMPLETE"}
TERMINAL_BAD = {"ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"}


# --- the kernel script -----------------------------------------------------

def test_kernel_script_is_valid_plain_python():
    """No preprocessing. A script kernel is not a notebook."""
    src = KERNEL.read_text()
    try:
        ast.parse(src)
    except SyntaxError as exc:
        pytest.fail(f"line {exc.lineno}: {exc.msg}\n  {(exc.text or '').rstrip()}")


def test_kernel_script_has_no_notebook_magics():
    bad = [(i, ln.strip()) for i, ln in enumerate(KERNEL.read_text().splitlines(), 1)
           if ln.strip().startswith(("!", "%%")) or re.match(r"^%\w", ln.strip())]
    assert not bad, f"IPython magics are a SyntaxError in a script kernel: {bad}"


def test_kernel_metadata_and_script_agree():
    """kernel-metadata.json is generated, so check the generator's intent."""
    src = (ROOT / "scripts" / "package_for_kaggle.py").read_text()
    assert '"kernel_type": "script"' in src
    assert '"code_file": "01_sft.py"' in src, \
        "the metadata must name the file these tests check"


# --- the poller ------------------------------------------------------------

def _classify(message: str) -> str:
    """Run the workflow's own sed+case against one status message."""
    shell = r'''
        token=$(printf '%s' "$1" | sed -n 's/.*KernelWorkerStatus\.\([A-Z_]*\).*/\1/p')
        case "${token:-}" in
          COMPLETE) echo DONE ;;
          ERROR|CANCEL_REQUESTED|CANCEL_ACKNOWLEDGED) echo FAILED ;;
          QUEUED|RUNNING|NEW_SCRIPT) echo WAIT ;;
          *) echo UNPARSED ;;
        esac
    '''
    r = subprocess.run(["bash", "-c", shell, "_", message],
                       capture_output=True, text=True, timeout=20)
    return r.stdout.strip()


@pytest.mark.parametrize("status", STATUSES)
def test_every_real_status_is_classified(status):
    msg = f'nimbou/manim-forge-sft has status "KernelWorkerStatus.{status}"'
    got = _classify(msg)
    want = ("DONE" if status in TERMINAL_OK else
            "FAILED" if status in TERMINAL_BAD else "WAIT")
    assert got == want, f"{status} classified as {got}, expected {want}"


def test_the_workflow_uses_this_exact_logic():
    """A test of a copy is a test of nothing."""
    wf = WORKFLOW.read_text()
    assert "KernelWorkerStatus" in wf, "the poller no longer extracts a token"
    for s in TERMINAL_BAD:
        assert s in wf, f"{s} is not handled in the workflow"
    assert "COMPLETE)" in wf


def test_a_slug_containing_a_keyword_is_not_a_status():
    """The old matcher globbed the whole message, slug included."""
    assert _classify(
        'nimbou/run-to-completion has status "KernelWorkerStatus.RUNNING"') == "WAIT"


@pytest.mark.parametrize("junk", ["", "401 Unauthorized", "connection reset"])
def test_unreadable_output_is_not_mistaken_for_a_verdict(junk):
    """A transient CLI failure must not read as success or as failure."""
    assert _classify(junk) == "UNPARSED"


def test_consecutive_unreadable_statuses_eventually_fail():
    wf = WORKFLOW.read_text()
    assert "unparsed" in wf, "no counter for unreadable statuses"
    assert re.search(r'unparsed.*-ge\s*\d+', wf), \
        "unreadable statuses must fail the run after a few in a row"


# --- the dataset layout Kaggle actually produces ---------------------------

def _forge_path_fn():
    """Pull _add_forge_to_path out of the kernel script and make it callable."""
    import re
    src = KERNEL.read_text()
    m = re.search(r'def _add_forge_to_path.*?\n(?=\nprint)', src, re.S)
    assert m, "the kernel no longer defines _add_forge_to_path"
    return m.group(0)


@pytest.mark.parametrize("layout,build", [
    # What Kaggle really does: it unpacks forge.tar.gz on upload, so the
    # package arrives as forge/forge/. Run 3 died opening the tarball that
    # no longer existed, one line before it would have loaded any data.
    ("kaggle-extracted", lambda d: (d / "forge" / "forge").mkdir(parents=True)),
    ("flat", lambda d: (d / "forge").mkdir(parents=True)),
])
def test_forge_is_found_in_every_layout_kaggle_produces(tmp_path, layout, build):
    import sys as _sys, tarfile as _tar
    data, work = tmp_path / "in", tmp_path / "work"
    data.mkdir(); work.mkdir()
    build(data)
    pkg = next(p for p in data.rglob("forge") if p.is_dir() and
               not (p / "forge").exists())
    (pkg / "__init__.py").touch()

    ns = {"DATA": data, "WORK": work, "sys": _sys, "tarfile": _tar,
          "Path": type(tmp_path)}
    exec(_forge_path_fn(), ns)
    before = list(_sys.path)
    try:
        msg = ns["_add_forge_to_path"]()
        assert "importable" in msg, msg
        root = Path(_sys.path[0])
        assert (root / "forge" / "__init__.py").exists(), \
            f"{root} is not the package's parent"
    finally:
        _sys.path[:] = before


def test_a_missing_forge_package_does_not_kill_the_run(tmp_path):
    """SFT never imports forge. It must not die because forge is absent."""
    import sys as _sys, tarfile as _tar
    data, work = tmp_path / "in", tmp_path / "work"
    data.mkdir(); work.mkdir()
    ns = {"DATA": data, "WORK": work, "sys": _sys, "tarfile": _tar,
          "Path": type(tmp_path)}
    exec(_forge_path_fn(), ns)
    msg = ns["_add_forge_to_path"]()
    assert "not found" in msg and "SFT" in msg


def test_sft_does_not_import_forge():
    """If this ever changes, the tolerant path handling above is not enough."""
    import ast
    tree = ast.parse(KERNEL.read_text())
    for n in ast.walk(tree):
        mod = (n.module if isinstance(n, ast.ImportFrom) else None)
        names = ([a.name for a in n.names]
                 if isinstance(n, (ast.Import, ast.ImportFrom)) else [])
        assert not (mod or "").startswith("forge"), \
            "SFT now imports forge — make the package a hard requirement"
        assert not any(x.startswith("forge") for x in names), \
            "SFT now imports forge — make the package a hard requirement"
