"""Can a Kaggle session render Manim? GRPO's reward is "the beat renders",
so the trainer needs a renderer beside the GPU. CPU session: no GPU quota."""
import json, os, subprocess, sys, time, textwrap

def sh(cmd, t=1800):
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)
    return r.returncode, time.time() - t0, (r.stdout + r.stderr)[-1500:]

out = {"nproc": os.cpu_count()}
out["apt"] = sh("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
                "libcairo2-dev libpango1.0-dev ffmpeg texlive-latex-base texlive-latex-extra "
                "texlive-fonts-recommended dvisvgm > /dev/null")[:2]
out["pip"] = sh(f"{sys.executable} -m pip install -q manim==0.21.0")[:2]

scene = textwrap.dedent('''
    from manim import *
    class T(Scene):
        def construct(self):
            c = Circle(color=BLUE); t = Text("hello"); m = MathTex(r"\\\\pi r^2")
            self.play(Create(c)); self.play(Write(t)); t.next_to(c, DOWN)
            self.play(Write(m.next_to(t, DOWN))); self.wait(1)
''')
open("t.py", "w").write(scene)
times = []
for i in range(3):
    code, dt, log = sh(f"{sys.executable} -m manim -ql --disable_caching t.py T", 600)
    times.append(round(dt, 1))
    if code:
        out["render_log"] = log
        break
out["render_rc"] = code
out["render_s"] = times
# Parallel: 4 at once, the shape GRPO needs (a group of completions per prompt)
t0 = time.time()
ps = [subprocess.Popen(f"{sys.executable} -m manim -ql --disable_caching --media_dir m{i} t.py T",
                       shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for i in range(4)]
out["parallel4_rc"] = [p.wait() for p in ps]
out["parallel4_s"] = round(time.time() - t0, 1)
print(json.dumps(out, indent=1))
json.dump(out, open("/kaggle/working/probe.json", "w"), indent=1)
