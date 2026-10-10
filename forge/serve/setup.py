"""Download what the app needs and check that it all works, once.

    ./.venv/bin/python -m forge.serve.setup

Run by install.sh. Prints one line per check, and what to do if one fails.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

OK, BAD = "  ✓", "  ✗"


def check(label: str, fn) -> bool:
    t0 = time.time()
    try:
        extra = fn() or ""
        print(f"{OK} {label}{(' — ' + extra) if extra else ''} ({time.time() - t0:.0f} s)", flush=True)
        return True
    except Exception as exc:                                  # noqa: BLE001
        print(f"{BAD} {label}: {exc}", flush=True)
        return False


def tools():
    missing = [t for t in ("ffmpeg", "latex", "dvisvgm") if shutil.which(t) is None]
    if missing:
        raise RuntimeError("not found: " + ", ".join(missing)
                           + " (run ./install.sh again, then open a new Terminal window)")


def model():
    from huggingface_hub import snapshot_download
    from forge.app.oneshot import BASE_MODEL
    snapshot_download(BASE_MODEL)
    return BASE_MODEL.split("/")[-1]


def embedder():
    from huggingface_hub import snapshot_download
    from forge.kit.library import EMBED_MODEL
    snapshot_download(EMBED_MODEL)


def voice():
    # In a child process: the speech engine's native code can kill the
    # process outright (a long espeak data path did, 2026-10-10).
    code = ("from pathlib import Path; import tempfile; from forge.app import voice as v; "
            "g = v.speak(['Ready.'], Path(tempfile.mkdtemp())); "
            "raise SystemExit(0 if g and g[0][0] else 3)")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600,
                       cwd=Path(__file__).resolve().parents[2])
    if p.returncode != 0:
        tail = [x for x in (p.stderr or "").splitlines() if x.strip()][-1:] or ["no output"]
        raise RuntimeError("narration unavailable, videos will be silent (" + tail[0][:160] + ")")


def render():
    from forge.app.twostage import PREAMBLE, kit_source
    body = '''
        stage = Stage(self)
        stage.title("Manim Forge works")
        ax = draw_axes(stage, x_range=(0, 4), y_range=(0, 9))
        plot_graph(stage, ax, lambda x: x ** 2, label="x^2")
        stage.equation(r"3^2 = 9")
        stage.mark()
'''
    src = PREAMBLE + "\n" + kit_source() + "\nclass Check(Scene):\n    def construct(self):" + body
    d = Path(tempfile.mkdtemp())
    (d / "check.py").write_text(src)
    p = subprocess.run([sys.executable, "-m", "manim", "-ql", "--disable_caching",
                        "-o", "check.mp4", "check.py", "Check"], cwd=d,
                       capture_output=True, text=True, timeout=300)
    if p.returncode != 0 or not list(d.rglob("check.mp4")):
        raise RuntimeError("a test scene did not render:\n" + (p.stderr or p.stdout)[-800:])


def main() -> int:
    print("Checking Manim Forge (the first run downloads about 6 GB)…", flush=True)
    results = [check("drawing tools (ffmpeg, LaTeX)", tools),
               check("AI model", model),
               check("scene search model", embedder),
               check("voice", voice),
               check("a test animation", render)]
    if all(results):
        print("\nAll set. Double-click “Start Manim Forge” to begin.")
        return 0
    if all(results[:2]) and results[4]:
        print("\nReady, with the problems above (the app will still work).")
        return 0
    print("\nSomething is missing; see the ✗ lines above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
