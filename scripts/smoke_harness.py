"""Phase 0 acceptance test: prove the harness works before anything depends on it."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.harness import RenderHarness, ErrorKind, find_scene_classes

CASES = {
    "valid_no_latex": '''
from manim import *
class Good(Scene):
    def construct(self):
        c = Circle(radius=1.5, color=BLUE)
        t = Text("forge", font_size=36)
        self.play(Create(c)); self.play(Write(t)); self.wait(0.1)
''',
    "syntax_error": '''
from manim import *
class Broken(Scene)
    def construct(self):
        pass
''',
    "api_misuse": '''
from manim import *
class Bad(Scene):
    def construct(self):
        self.play(Create(Circle(shade_of_purple=3)))
''',
    "name_error": '''
from manim import *
class Undef(Scene):
    def construct(self):
        self.play(Create(Hexahedron()))
''',
    "no_scene": '''
from manim import *
def helper(): return 42
''',
    "manimgl_import": '''
from manimlib import *
class GLScene(Scene):
    def construct(self):
        self.play(ShowCreation(Circle()))
''',
    "needs_latex": '''
from manim import *
class Tex1(Scene):
    def construct(self):
        self.play(Write(MathTex(r"\\sum_{n=1}^{\\infty} \\frac{1}{n^2}")))
''',
}

h = RenderHarness(python_bin="./.venv/bin/python", cache_dir="data/frames", timeout=120)

print(f"{'case':<18} {'ok':<6} {'error_kind':<16} {'env?':<6} {'frames':<7} {'sec'}")
print("-" * 68)
t0 = time.monotonic()
results = {}
for name, code in CASES.items():
    r = h.render(code, quality="low", frames=4)
    results[name] = r
    print(f"{name:<18} {str(r.ok):<6} {r.error_kind.value:<16} "
          f"{str(r.is_environment_failure):<6} {len(r.frame_paths):<7} {r.elapsed_s:.1f}")
print("-" * 68)
print(f"total {time.monotonic()-t0:.1f}s")

print("\nAST scene detection:", find_scene_classes(CASES['valid_no_latex']))
print("\nRepair feedback for api_misuse:\n" + results['api_misuse'].feedback(6))

# cache check
t = time.monotonic(); r2 = h.render(CASES['valid_no_latex'], quality='low', frames=4)
print(f"\ncache hit: from_cache={r2.from_cache} in {time.monotonic()-t:.3f}s")
